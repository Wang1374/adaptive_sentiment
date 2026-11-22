"""Train on MR / SST-2; save training curves every epoch and evaluate the test set with the best validation checkpoint."""

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data import DATASETS, DEFAULT_DATA_ROOT, BatchCollator, build_vocab, read_splits
from model import ROUTES, AdaptiveSentiment, training_loss


def choose_device(name):
    return torch.device(("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else name)


def run_epoch(model, loader, args, device, desc, optimizer=None):
    training = optimizer is not None
    model.train(training)
    # Accumulate on device to avoid forcing a GPU sync for every batch's metrics.
    total = torch.zeros(2 + len(ROUTES), device=device)
    count = 0
    bar = tqdm(loader, desc=desc, unit="batch", dynamic_ncols=True)
    with torch.set_grad_enabled(training):
        for step, (ids, mask, labels) in enumerate(bar, 1):
            ids, mask, labels = (t.to(device, non_blocking=True) for t in (ids, mask, labels))
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits, routes, all_scores = model(ids, mask, args.tau)
            loss = training_loss(logits, all_scores, labels, args.aux_weight, args.calibration_weight)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
                optimizer.step()
            count += labels.numel()
            total[0] += loss.detach() * labels.numel()
            total[1] += logits.detach().argmax(-1).eq(labels).sum()
            total[2:] += torch.stack([(routes == r).sum() for r in range(len(ROUTES))])
            if step % 10 == 0 or step == len(loader):
                avg_loss, accuracy = (total[:2] / count).tolist()
                bar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{accuracy:.1%}")
    return dict(zip(("loss", "accuracy", *ROUTES), (total / count).tolist()))


@torch.inference_mode()
def evaluate_test(model, loader, args, device):
    model.eval()
    counts = torch.zeros(2 * len(ROUTES), dtype=torch.long, device=device)
    confusion = torch.zeros(4, dtype=torch.long, device=device)
    for ids, mask, labels in tqdm(loader, desc="Test (sparse routing)", unit="batch"):
        ids, mask, labels = (t.to(device, non_blocking=True) for t in (ids, mask, labels))
        logits, route = model.predict(ids, mask, args.tau)
        predictions = logits.argmax(-1)
        correct = predictions.eq(labels)
        counts[:len(ROUTES)] += torch.stack([(route == r).sum() for r in range(len(ROUTES))])
        counts[len(ROUTES):] += torch.stack([((route == r) & correct).sum() for r in range(len(ROUTES))])
        confusion += torch.stack([((labels * 2 + predictions) == r).sum() for r in range(4)])
    tn, fp, fn, tp = confusion.tolist()
    f1 = lambda t, f: 2 * t / (2 * t + f) if 2 * t + f else 0.0
    total = tn + fp + fn + tp
    values = counts.tolist()
    return {
        "accuracy": (tp + tn) / total,
        "macro_f1": (f1(tp, fp + fn) + f1(tn, fp + fn)) / 2,
        "confusion_true_rows_pred_cols": [[tn, fp], [fn, tp]],
        "routes": {name: {"count": n, "fraction": n / total, "accuracy": c / n if n else None}
                   for name, n, c in zip(ROUTES, values[:len(ROUTES)], values[len(ROUTES):])},
    }


def save_history(out, history, dataset):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MaxNLocator

    with (out / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    epochs = [r["epoch"] for r in history]
    ax.plot(epochs, [r["train_loss"] for r in history], "o-", color="#176C81", label="Train")
    ax.plot(epochs, [r["val_loss"] for r in history], "s-", color="#B85827", label="Validation")
    ax.set(title=f"{dataset.upper()} | Training and validation loss", xlabel="Epoch", ylabel="Total loss")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", color="#E3E7EC")
    ax.legend(frameon=False)
    fig.savefig(out / "loss_curve.png", dpi=180)
    plt.close(fig)


def setup_encoder(args, train, out):
    if args.encoder == "scratch":
        vocab = build_vocab(train, args.max_vocab, args.min_freq)
        (out / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
        return BatchCollator(args.max_length, vocab=vocab), len(vocab), None
    from transformers import AutoModel, AutoTokenizer

    options = dict(local_files_only=not args.allow_download, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, **options)
    pretrained = AutoModel.from_pretrained(args.model_name_or_path, **options)
    model_limit = min(tokenizer.model_max_length, getattr(pretrained.config, "max_position_embeddings", 10**9))
    if args.max_length > model_limit:
        raise ValueError(f"max_length={args.max_length} exceeds encoder limit {model_limit}")
    tokenizer.save_pretrained(out / "tokenizer")
    pretrained.config.save_pretrained(out / "encoder_config")
    return BatchCollator(args.max_length, tokenizer=tokenizer), None, pretrained


def run_dataset(args, dataset, device):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    splits = read_splits(args.data_root, dataset, args.split_dir)
    out = Path(args.output_dir) / dataset
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} already contains a run. Choose a fresh --output-dir.")
    out.mkdir(parents=True, exist_ok=True)
    collate, vocab_size, pretrained = setup_encoder(args, splits["train"], out)
    model_args = {key: getattr(args, key) for key in ("max_length", "dim", "trans_dim", "heads", "depth", "dropout")}
    model = AdaptiveSentiment(**model_args, vocab_size=vocab_size, pretrained=pretrained).to(device)
    loaders = {name: DataLoader(rows, batch_size=args.batch_size, shuffle=name == "train",
                                collate_fn=collate, pin_memory=device.type == "cuda",
         
                                generator=torch.Generator().manual_seed(args.seed))
               for name, rows in splits.items()}
    encoder_group = list(model.encoder.parameters()) if pretrained is not None else []
    encoder_ids = {id(p) for p in encoder_group}
    groups = [{"params": [p for p in model.parameters() if id(p) not in encoder_ids], "lr": args.lr}]
    if encoder_group:
        groups.append({"params": encoder_group, "lr": args.encoder_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    config = vars(args) | {"dataset": dataset, "device_used": str(device), "torch_version": str(torch.__version__),
                           "split_sizes": {k: len(v) for k, v in splits.items()},
                           "split_labels": {k: dict(Counter(y for _, y in v)) for k, v in splits.items()},
                           "model_args": model_args, "vocab_size": vocab_size}
    (out / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{dataset}: {config['split_sizes']}, encoder={args.encoder}, device={device}", flush=True)
    best_key, best_epoch, history = None, None, []
    checkpoint = out / "best_model.pt"
    for epoch in range(1, args.epochs + 1):
        prefix = f"{dataset} {epoch:02d}/{args.epochs:02d}"
        train = run_epoch(model, loaders["train"], args, device, prefix + " train", optimizer)
        val = run_epoch(model, loaders["validation"], args, device, prefix + " val  ")
        history.append({"epoch": epoch, **{f"train_{k}": v for k, v in train.items()},
                        **{f"val_{k}": v for k, v in val.items()}})
        key = (val["accuracy"], -val["loss"])
        if best_key is None or key > best_key:
            best_key, best_epoch = key, epoch
            torch.save(model.state_dict(), checkpoint)
        torch.save(model.state_dict(), out / "last_model.pt")  # Overwritten every epoch; holds the final epoch's weights at the end
        save_history(out, history, dataset)
        pprint = ", ".join(f"{k}={val[k]:.1%}" for k in ROUTES)
        tqdm.write(f"{prefix} train loss={train['loss']:.4f}  val loss={val['loss']:.4f} "
                   f"acc={val['accuracy']:.2%}  val routes: {pprint}")
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    metrics = evaluate_test(model, loaders["test"], args, device)
    metrics.update(best_epoch=best_epoch, best_val_accuracy=best_key[0], best_val_loss=-best_key[1], tau=args.tau)
    (out / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"{dataset} best epoch={best_epoch} test accuracy={metrics['accuracy']:.2%} "
          f"macro-F1={metrics['macro_f1']:.2%} | {out.resolve()}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--split-dir", default="prepared_seed42")
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "runs"))
    parser.add_argument("--encoder", choices=("scratch", "pretrained"), default="scratch")
    parser.add_argument("--model-name-or-path", default="distilbert/distilbert-base-uncased")
    parser.add_argument("--allow-download", action="store_true", help="Allow Hugging Face model downloads")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--max-vocab", type=int, default=20000)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--trans-dim", type=int, default=48)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--tau", type=float, default=0.9)
    parser.add_argument("--aux-weight", type=float, default=0.4)
    parser.add_argument("--calibration-weight", type=float, default=0.1)
    a = parser.parse_args()
    if min(a.epochs, a.batch_size, a.cpu_threads, a.heads, a.depth, a.min_freq) < 1:
        parser.error("epochs, batch-size, cpu-threads, heads, depth and min-freq must be positive")
    if a.max_length < 3 or a.max_vocab < 5 or not 2 <= a.trans_dim < a.dim:
        parser.error("require max-length >= 3, max-vocab >= 5, and 2 <= trans-dim < dim")
    if a.dim % a.heads or a.trans_dim % a.heads:
        parser.error("dim and trans-dim must be divisible by heads")
    if not 0.5 <= a.tau <= 1.0 or not 0 <= a.dropout < 1:
        parser.error("tau must be in [0.5, 1.0] and dropout in [0, 1)")
    if min(a.aux_weight, a.calibration_weight, a.weight_decay) < 0 or min(a.lr, a.encoder_lr) <= 0:
        parser.error("loss weights and weight-decay must be nonnegative; learning rates must be positive")
    return a


def main():
    args = parse_args()
    device = choose_device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu or install GPU-enabled PyTorch")
    torch.set_num_threads(args.cpu_threads)
    for dataset in DATASETS if args.dataset == "all" else (args.dataset,):
        run_dataset(args, dataset, device)


if __name__ == "__main__":
    main()
