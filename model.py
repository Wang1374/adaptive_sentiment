"""TextCNN, TSSA and CAA over text; per-sample confidence routing at inference."""

import torch
from torch import nn
from torch.nn import functional as F


ROUTES = ("cnn", "transformer", "fusion")


def zero_pad(x, mask):
    return x.masked_fill(~mask.unsqueeze(-1), 0)


def confidence(logits):
    return logits.float().softmax(dim=-1).amax(dim=-1)


class ScratchEncoder(nn.Module):
    """Small bidirectional BERT-style encoder that runs offline; weights start from random initialization."""

    def __init__(self, vocab_size, dim, heads, max_length, dropout):
        super().__init__()
        self.word = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.pos = nn.Embedding(max_length, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.layer = nn.TransformerEncoderLayer(
            dim, heads, 4 * dim, dropout, activation="gelu", batch_first=True
        )

    def forward(self, ids, mask):
        pos = torch.arange(ids.size(1), device=ids.device)
        x = self.dropout(self.norm(self.word(ids) + self.pos(pos)))
        return self.layer(x, src_key_padding_mask=~mask)


class TextCNN(nn.Module):
    """Conv1d slides along the token axis only; each kernel size captures local patterns of a different span."""

    def __init__(self, dim, dropout, kernels=(3, 5, 7)):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(dim, dim // 2, k, padding=k // 2) for k in kernels])
        self.mix = nn.Linear(len(kernels) * (dim // 2), dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        xt = zero_pad(x, mask).transpose(1, 2)  # [B, D, L]
        local = torch.cat([F.gelu(conv(xt)) for conv in self.convs], dim=1).transpose(1, 2)
        return zero_pad(self.norm(x + self.dropout(self.mix(local))), mask)


class AttentionTSSA(nn.Module):
    """TSSA from module 129 with a padding mask; never builds an [L, L] token similarity matrix."""

    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim, bias=False)  # The original operator uses a single projection, not separate Q/K/V.
        self.temp = nn.Parameter(torch.ones(heads, 1))
        self.attn_drop = nn.Dropout(dropout)
        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

    def forward(self, x, mask):
        b, length, dim = x.shape
        w = self.qkv(x).reshape(b, length, self.heads, dim // self.heads).transpose(1, 2)
        valid = mask[:, None, :, None]
        w_valid = w.float().masked_fill(~valid, 0)
        w_unit = F.normalize(w_valid, dim=-2)
        # Softmax over heads for each token; normalizing over tokens is only used to estimate per-head second moments.
        pi = (w_unit.square().sum(dim=-1) * self.temp.float()).softmax(dim=1)
        pi = pi.masked_fill(~valid.squeeze(-1), 0)
        token_weights = pi / pi.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        dots = (token_weights.unsqueeze(-1) * w_valid.square()).sum(dim=-2, keepdim=True)
        attn = self.attn_drop((1 + dots).reciprocal())
        out = (-w_valid * pi.unsqueeze(-1) * attn).transpose(1, 2).reshape(b, length, dim)
        return zero_pad(self.to_out(out.to(w.dtype)), mask)


class TSSABlock(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = AttentionTSSA(dim, heads, dropout)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(4 * dim, dim), nn.Dropout(dropout))

    def forward(self, x, mask):
        x = x + self.attn(self.norm1(x), mask)
        return zero_pad(x + self.ffn(self.norm2(x)), mask)


class ContextAnchorAttention1D(nn.Module):
    """1D version of module 083: local average -> pointwise projection -> two depthwise convs -> pointwise projection -> gate."""

    def __init__(self, dim):
        super().__init__()
        self.in_proj = nn.Linear(2 * dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.near = nn.Conv1d(dim, dim, 7, padding=3, groups=dim, bias=False)
        self.far = nn.Conv1d(dim, dim, 7, padding=6, dilation=2, groups=dim, bias=False)
        self.out_proj = nn.Conv1d(dim, dim, 1)

    def forward(self, cnn, transformer, mask):
        valid = mask.unsqueeze(1)
        x = zero_pad(torch.cat((cnn, transformer), dim=-1), mask).transpose(1, 2)
        # Pool the count of valid positions too, so trailing padding does not drag down the local average.
        numerator = F.avg_pool1d(x, 7, stride=1, padding=3)
        denominator = F.avg_pool1d(valid.to(x.dtype), 7, stride=1, padding=3)
        pooled = numerator / denominator.clamp_min(1 / 7)
        x = F.silu(self.norm(self.in_proj(pooled.transpose(1, 2)))).transpose(1, 2)
        x = self.near(x.masked_fill(~valid, 0)).masked_fill(~valid, 0)
        x = self.far(x)
        gate = self.out_proj(x).transpose(1, 2).sigmoid()  # [B, L, D], fused per position and per channel.
        return zero_pad(gate * cnn + (1 - gate) * transformer, mask)


class AdaptiveSentiment(nn.Module):
    def __init__(self, *, max_length, dim, trans_dim, heads, depth, dropout,
                 vocab_size=None, pretrained=None):
        super().__init__()
        if pretrained is None:
            self.encoder = ScratchEncoder(vocab_size, dim, heads, max_length, dropout)
            encoder_dim = dim
        else:
            self.encoder = pretrained
            encoder_dim = pretrained.config.hidden_size
        self.is_pretrained = pretrained is not None
        self.project = nn.Identity() if encoder_dim == dim else nn.Linear(encoder_dim, dim)
        self.cnn = TextCNN(dim, dropout)
        self.lower = nn.Linear(dim, trans_dim)
        self.tssa_blocks = nn.ModuleList(TSSABlock(trans_dim, heads, dropout) for _ in range(depth))
        self.upper = nn.Linear(trans_dim, dim)
        self.fusion = ContextAnchorAttention1D(dim)
        # All three branches share one head, aligning their output space so confidence uses the same softmax.
        self.head = nn.Sequential(nn.LayerNorm(2 * dim), nn.Dropout(dropout), nn.Linear(2 * dim, 2))

    def encode(self, ids, mask):
        if self.is_pretrained:
            x = self.encoder(input_ids=ids, attention_mask=mask.long()).last_hidden_state
        else:
            x = self.encoder(ids, mask)
        return zero_pad(self.project(x), mask)  # [B, L, D]

    def transform(self, x, mask):
        x = zero_pad(self.lower(x), mask)  # [B, L, trans_dim]
        for block in self.tssa_blocks:
            x = block(x, mask)
        return zero_pad(self.upper(x), mask)

    def classify(self, x, mask):
        mean = zero_pad(x, mask).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        maximum = x.masked_fill(~mask.unsqueeze(-1), torch.finfo(x.dtype).min).amax(dim=1)
        return self.head(torch.cat((mean, maximum), dim=-1))  # [B, 2] raw logits

    def forward(self, ids, mask, tau):
        """Full forward pass over all branches for train/val loss; hard routing picks the main-task logits, auxiliary losses train every branch."""
        x = self.encode(ids, mask)
        cnn = self.cnn(x, mask)
        trans = self.transform(x, mask)
        scores = (self.classify(cnn, mask), self.classify(trans, mask),
                  self.classify(self.fusion(cnn, trans, mask), mask))
        c_cnn, c_trans = (confidence(z.detach()) for z in scores[:2])
        routes = torch.where(c_cnn >= tau, 0, torch.where(c_trans >= c_cnn, 1, 2))
        logits = torch.stack(scores, dim=1)[torch.arange(ids.size(0), device=ids.device), routes]
        return logits, routes, scores

    @torch.no_grad()
    def predict(self, ids, mask, tau):
        """Sparse inference: only samples that do not exit early go through TSSA and CAA; original sample order is restored."""
        x = self.encode(ids, mask)
        cnn = self.cnn(x, mask)
        logits = self.classify(cnn, mask)
        c_cnn = confidence(logits)
        routes = torch.zeros_like(c_cnn, dtype=torch.long)
        deep = (c_cnn < tau).nonzero(as_tuple=True)[0]
        if deep.numel():
            deep_mask = mask[deep]
            trans = self.transform(x[deep], deep_mask)
            trans_logits = self.classify(trans, deep_mask)
            logits[deep] = trans_logits
            routes[deep] = 1
            needs_fusion = confidence(trans_logits) < c_cnn[deep]
            fuse = needs_fusion.nonzero(as_tuple=True)[0]
            if fuse.numel():
                dest = deep[fuse]
                fused = self.fusion(cnn[dest], trans[fuse], mask[dest])
                logits[dest] = self.classify(fused, mask[dest])
                routes[dest] = 2
        return logits, routes


def training_loss(logits, all_scores, labels, aux_weight, calibration_weight):
    """Binary task with reported confidence uses CE + softmax; the sigmoid in CAA is only a feature gate."""
    main_ce = F.cross_entropy(logits, labels)
    aux_ce = torch.stack([F.cross_entropy(z, labels) for z in all_scores]).mean()
    target = F.one_hot(labels, 2)
    brier = torch.stack([(z.float().softmax(-1) - target).square().sum(-1).mean()
                         for z in all_scores]).mean()
    return main_ce + aux_weight * aux_ce + calibration_weight * brier
