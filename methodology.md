# Methodology

## 1. Research objective and scope

This study investigates confidence-aware conditional computation for binary English sentiment classification. The predictive research question is whether a compact model can reserve additional feature extraction for examples with lower initial confidence while retaining useful classification performance on MR and SST-2.

The framework combines a shared scratch Transformer encoder, a TextCNN branch, a reduced-dimension TSSA branch, and a one-dimensional CAA fusion module. Its motivation is to balance classification quality and computational cost. The current experiments measure predictive quality and route coverage; a reduction in wall-clock latency is a hypothesis that still requires a matched benchmark.

The implementation does not directly measure semantic complexity. Confidence is a model-dependent routing signal. Statements that short or simple sentences always take the CNN route, or that difficult sentences necessarily require TSSA, are not established by the current experiments.

## 2. Data

### 2.1 Observation unit and labels

Each observation is a text record with one binary sentiment label: `0` for negative and `1` for positive. Inputs are stored in CSV files with `text` and `label` columns. The loader rejects empty text, labels other than `0` or `1`, missing required columns, and empty split files. It does not silently remove invalid observations.

### 2.2 Supplied splits

| Dataset | Training negative / positive | Validation negative / positive | Test negative / positive | Total records |
| --- | ---: | ---: | ---: | ---: |
| MR | 4,265 / 4,265 | 533 / 533 | 533 / 533 | 10,662 |
| SST-2 | 3,304 / 3,605 | 428 / 444 | 912 / 909 | 9,602 |

Thus MR has 8,530 training, 1,066 validation, and 1,066 test records. SST-2 has 6,909 training, 872 validation, and 1,821 test records.

The files are read from `<data-root>/<dataset>/prepared_seed42/`. The training code preserves the supplied split assignments and does not generate them. The name `prepared_seed42` is not a substitute for a recorded split-generation procedure.

The local data workflow references [SentiCSE](https://aclanthology.org/2024.lrec-main.1280/) and its [repository](https://github.com/nayohan/SentiCSE). However, the local SST-2 training count is 6,909, whereas Table 1 of that paper reports 67,349. The current protocol must be described using its actual local counts. An exact upstream dataset revision, acquisition date, split-generation script, and file checksums were not recorded with these experiments. These should be documented before claiming complete data reproducibility or exact benchmark equivalence.

### 2.3 Data integrity checks

A read-only audit of the locally available splits found no repeated texts within or across training, validation, and test sets after lowercasing and normalization with the current tokenizer. The saved vocabularies exactly match vocabularies reconstructed from the respective training sets. This checks normalized exact overlap, not near duplicates, shared authors, shared source documents, or other possible dependence between observations.

## 3. Preprocessing and batching

Scratch-mode processing is implemented in `data.py`:

1. Read CSV files using UTF-8 with optional byte-order-mark handling.
2. Lowercase text.
3. Tokenize words, internal apostrophes, and punctuation using `TOKEN_PATTERN`. Punctuation is retained as separate tokens.
4. Count tokens using the training split only. Retain tokens meeting `min_freq=1`, sorted by descending frequency with lexical tie-breaking.
5. Reserve `[PAD]`, `[UNK]`, `[CLS]`, and `[SEP]` at IDs 0, 1, 2, and 3. Cap the vocabulary at 20,000 entries including these tokens.
6. Truncate content to `max_length - 2` tokens, add `[CLS]` and `[SEP]`, and map unknown tokens to `[UNK]`.
7. Dynamically pad each batch to its longest sequence, and build a Boolean validity mask from nonzero IDs.

The selected maximum sequence length is 64, including the two boundary tokens. The resulting vocabulary sizes are 17,089 for MR and 13,846 for SST-2. There is no stemming, stop-word removal, numerical feature scaling, imputation, or sentiment-lexicon feature engineering in the scratch pipeline.

A local audit found no MR examples exceeding the 62-token content allowance and one such training example in SST-2; none of the SST-2 validation or test examples exceeded it under this tokenizer. These observations describe the supplied local files, not the original datasets generally.

## 4. Model architecture

Let `B` be batch size, `L` padded sequence length, `D` shared feature dimension, and `M[b,t]` the validity mask. `zero_pad` replaces feature values at invalid positions with zero.

### 4.1 Shared scratch encoder

The selected experiments use `encoder=scratch`:

```text
H0 = Dropout(LayerNorm(word_embedding(ids) + position_embedding(positions)))
X  = zero_pad(TransformerEncoderLayer(H0, padding_mask), M)
```

The shared encoder contains one standard bidirectional Transformer encoder layer with feed-forward width `4D`, GELU activation, and the configured dropout. `X` has shape `[B, L, D]`. No pretrained BERT or DistilBERT weights are used in the selected runs. The presence of a model-name argument in the saved configuration does not change this fact.

All inference examples execute this encoder. Its conventional self-attention still includes token-to-token interactions; replacing a later attention operator with TSSA does not eliminate that shared cost.

### 4.2 TextCNN branch

Three parallel one-dimensional convolutions act on the token axis with kernel sizes `3`, `5`, and `7`. Each convolution maps `D` channels to `floor(D/2)` channels and uses symmetric padding. Their GELU-activated outputs are concatenated, projected back to `D`, and combined with the shared representation:

```text
U = concatenate(GELU(Conv3(X)), GELU(Conv5(X)), GELU(Conv7(X)))
H_cnn = zero_pad(LayerNorm(X + Dropout(Linear(U))), M)
```

The convolutions are parallel, not a composition `Conv7(Conv5(Conv3(X)))`. Input padding is zeroed before convolution, and invalid output positions are masked afterward.

### 4.3 Reduced-dimension TSSA branch

TSSA receives `X`, not `H_cnn`. A linear layer reduces the feature dimension from `D` to `d=48`; two residual TSSA blocks operate at that width; a final linear layer restores `D`.

```text
Z0 = zero_pad(Linear_down(X), M)
Zk = TSSABlock_k(Zk-1, M), k = 1, 2
H_tssa = zero_pad(Linear_up(Z2), M)
```

Each block applies pre-normalized TSSA with a residual connection, followed by a pre-normalized feed-forward network `d -> 4d -> d`, GELU, dropout, and a second residual connection.

The implemented TSSA operator uses one learned feature projection, even though its attribute is named `qkv`. It does not construct separate query, key, and value projections or an `L x L` similarity matrix. For each head `h`, token `t`, and within-head feature `j`:

```text
W = reshape_into_heads(Linear(Z))
V = zero_invalid_positions(W)
N = L2_normalize(V, over=token_positions)
pi[h,t] = softmax_over_heads(temp[h] * sum_j(N[h,t,j]^2))
pi[h,t] = 0 at invalid token positions
alpha[h,t] = pi[h,t] / sum_valid_t(pi[h,t])
s[h,j] = sum_valid_t(alpha[h,t] * V[h,t,j]^2)
Y[h,t,j] = -V[h,t,j] * pi[h,t] / (1 + s[h,j])
```

The implementation applies dropout to the reciprocal factor, merges heads, applies an output projection and dropout, and masks padding again. Normalization denominators are clamped for numerical safety.

**Single-head behavior matters for MR.** With one head, the softmax across heads is one at valid positions. Therefore `alpha` becomes uniform over valid tokens, and, before output projection and with dropout disabled, the operator reduces to:

```text
Y[t,j] = -V[t,j] / (1 + mean_valid_t(V[t,j]^2))
```

It retains sequence-dependent second-moment scaling, residual computation, and feed-forward processing. It does not learn a nontrivial head-selection distribution in this case, and the head-temperature parameter cannot affect that softmax. The selected SST-2 model uses two heads and does not have this single-head simplification.

The source comments attribute TSSA to module 129 and reference [ToST](https://github.com/RobinWu218/ToST). The current method should be described as an adaptation of that operator with masking and dimension reduction, without claiming the operator itself as an original contribution.

### 4.4 One-dimensional CAA fusion

CAA combines token-aligned CNN and TSSA representations. Concatenated features are locally averaged with a window of 7. Dividing pooled feature values by the pooled validity mask excludes trailing padding from the average. A linear projection, layer normalization, and SiLU activation follow.

The gate network then applies a depthwise convolution with kernel 7, another depthwise convolution with kernel 7 and dilation 2, and a pointwise projection. Invalid intermediate positions are zeroed before the second depthwise convolution. A sigmoid produces `G` with shape `[B, L, D]`:

```text
H_fusion = zero_pad(G * H_cnn + (1 - G) * H_tssa, M)
```

This is feature gating per token and channel. It is not the scalar sentiment probability used for routing. The code identifies its source as an adaptation of module 083; a complete bibliographic reference for that source is not recorded in the repository.

### 4.5 Shared classifier

For each branch representation `H`, masked mean pooling and masked maximum pooling are concatenated into a `2D` vector. One shared head applies layer normalization, dropout, and a linear map to two sentiment logits. Sharing the head aligns the output classes, but does not guarantee that different branches' confidence scores are calibrated or directly comparable in reliability.

## 5. Confidence-aware inference

For branch `r`, define confidence as:

```text
C_r = max(softmax(logits_r))
```

The selected experiments use `tau=0.9`. Per-example inference is:

```text
X = shared_encoder(text)
H_cnn = cnn(X)
if C_cnn >= tau:
    return CNN prediction
H_tssa = tssa(X)
if C_tssa >= C_cnn:
    return TSSA prediction
return prediction from CAA(H_cnn, H_tssa)
```

Only the CNN decision uses the absolute threshold. A TSSA prediction can be selected even when its confidence is below `tau`, provided it is at least the CNN confidence. The fusion decision follows the relative comparison and does not test whether the fused prediction is more confident or more accurate than either constituent branch.

`predict()` groups examples requiring additional computation, processes the relevant sub-batches, and restores their predictions to the original batch positions. TSSA and CAA are genuinely skipped for early-exit examples. This avoids later branch computation, while all model weights remain part of the stored model.

## 6. Training objective and optimization

### 6.1 Full computation during training

`forward()` evaluates CNN, TSSA, and CAA for every training and validation example. Detached CNN and TSSA confidence values determine which branch logits enter the main cross-entropy term. This discrete route is not trained through a differentiable routing loss.

Auxiliary losses train all branches even when they are not selected for the main task. For one-hot target `y` and branch probabilities `p_r`, the objective is:

```text
L = CE(logits_selected, y)
  + 0.4 * mean_r(CE(logits_r, y))
  + 0.1 * mean_r(mean_examples(sum_classes((p_r - y)^2)))
```

The last term is a multiclass Brier loss. It encourages probability accuracy but does not certify calibration. The branch mean includes all three branches. `--aux-weight` and `--calibration-weight` configure the two coefficients.

Training and validation total loss include the same three components. Test classification uses sparse routing and reports accuracy and Macro-F1 rather than that full training loss. No selective-computation training-time saving is established by this implementation.

### 6.2 Selected configurations

| Setting | MR | SST-2 |
| --- | ---: | ---: |
| Encoder | scratch | scratch |
| Shared feature dimension `D` | 64 | 96 |
| TSSA dimension `d` | 48 | 48 |
| Heads in shared encoder and TSSA | 1 | 2 |
| TSSA blocks | 2 | 2 |
| Maximum sequence length | 64 | 64 |
| Maximum vocabulary / minimum frequency | 20,000 / 1 | 20,000 / 1 |
| CNN kernels | 3, 5, 7 | 3, 5, 7 |
| Dropout | 0.5 | 0.5 |
| Batch size | 256 | 256 |
| Epoch budget | 100 | 100 |
| Optimizer | AdamW | AdamW |
| Learning rate | 0.001 | 0.001 |
| Weight decay | 0.01 | 0.01 |
| Gradient clipping norm | 1.0 | 1.0 |
| Routing threshold | 0.9 | 0.9 |
| Auxiliary / Brier weights | 0.4 / 0.1 | 0.4 / 0.1 |
| Random seed | 42 | 42 |
| Device / CPU threads | CPU / 8 | CPU / 8 |

The code seeds Python's random generator and PyTorch, and uses seeded data-loader generators. Training batches are shuffled; validation and test batches are not. The code does not enable every deterministic backend setting, so exact numerical reproduction across environments is not guaranteed. There is no learning-rate scheduler or early stopping in the scratch training loop; every selected run completed 100 epochs.

## 7. Validation and model selection

Within each run, validation is performed after every epoch with dropout disabled. The checkpoint selection key is `(validation_accuracy, -validation_loss)`. The selected checkpoint is reloaded after training and evaluated on the test set using sparse routing. The selected epochs are 95 for MR and 56 for SST-2.

Across experiments, the final MR and SST-2 source runs were selected by comparing test accuracy. This introduces test-informed model selection. The reported test values may therefore be optimistic, and should not be described as scores from a completely untouched final evaluation. Freezing these selected results does not remove their selection history.

The repository contains several manually configured experiments. It does not implement or log a systematic grid, random, or Bayesian hyperparameter search. The selected experiments use one seed and one fixed split per dataset. Cross-validation, repeated-seed estimates, and confidence intervals are not available from these records. Epoch-to-epoch variation is not a replacement for independent validation repetitions.

## 8. Evaluation measures

With negative class 0 and positive class 1, confusion matrices have true labels in rows and predicted labels in columns:

```text
[[TN, FP],
 [FN, TP]]
```

```text
Accuracy = (TN + TP) / (TN + FP + FN + TP)
F1_positive = 2TP / (2TP + FP + FN)
F1_negative = 2TN / (2TN + FP + FN)
Macro-F1 = (F1_positive + F1_negative) / 2
```

The primary reported measures are overall accuracy and Macro-F1. Macro-F1 weights the two classes equally. Route coverage is the number of examples assigned to a route divided by test size. Route accuracy uses only examples assigned to that route. These subsets differ in selection, so their accuracies cannot by themselves establish that one branch outperforms another on the same examples.

Parameter counts describe the tensor elements in the saved model state. Counts include embeddings and all branches, independent of the threshold and exit fractions. They are not FLOP counts or latency measurements.

## 9. Efficiency analysis boundaries

For an illustrative per-example cost model, let `C0` include shared encoding, CNN, and its classifier; `CT` include TSSA and its classifier; and `CF` include fusion and its classifier. Let `q` be the fraction reaching TSSA and `r` the fraction reaching CAA:

```text
Expected adaptive inference cost approximately equals C0 + q * CT + r * CF.
```

This is a conceptual accounting expression, not a measured runtime model. Actual cost depends on sequence lengths, batch composition, indexing overhead, device utilization, and implementation. A 75% early-exit share does not imply a 75% reduction in total time.

A controlled study would compare the fixed selected checkpoint under adaptive routing and always-compute execution on the same inputs, then compare separately trained baselines for predictive quality. Measurements should keep hardware, software, sequence lengths, batch size, CPU thread count, warm-up, synchronization, and timing boundaries consistent. Report latency and throughput separately from parameter counts. No such benchmark has yet been saved with the selected experiments.

## 10. Software and reproducibility

Saved configurations record PyTorch 2.2.2, CPU execution, and eight CPU threads. Current installation files instead require PyTorch 2.6 or later and constrain matplotlib and tqdm. The exact Python version and historical versions of the remaining libraries were not saved; Python 3.10.12 and any IDE or CPU model should not be asserted as measured experiment metadata without another record.

The provided checkpoints, vocabularies, configurations, histories, and metrics allow inspection and inference. Retraining also requires the local raw CSV splits. The current repository has no standalone command for reevaluating all test records from a checkpoint; `train.py` evaluates after training and `predict.py` accepts individual text arguments.

## 11. Relationship to the reference literature

| Reference | Relevant idea | Difference from the current implementation |
| --- | --- | --- |
| Bi and Zhang (2024), BCBA | BERT representations, parallel CNN/BiGRU extraction, attention fusion | Current selected runs use a small scratch encoder, TSSA instead of BiGRU, and conditional branch execution |
| Chen, Sun, and Yan (2023) | Parallel BERT-CNN-BiLSTM-attention sentiment classification | Uses a different recurrent branch and Chinese hotel-review data; published scores are not a matched MR/SST-2 baseline |
| Zhao, Liu, and Wang (2023), TETBA | ERNIE, parallel TextCNN/BiGRU, attention, source-to-target parameter transfer | Current runs use supervised scratch training and do not implement that transfer-learning procedure |
| Kim et al. (2024), SentiCSE | Sentiment-aware word/sentence pretraining and representation evaluation | Current code does not implement quadruplet contrastive learning, sentiment-aware MLM, SgTS, or Spearman-based checkpoint selection |

The research contribution under investigation is the combination of compact representations and confidence-based conditional execution. The present evidence does not establish priority over all prior adaptive-inference research, and does not isolate the contribution of each component without ablations.

## References

1. Bi, X., and Zhang, T. (2024). Pedagogical sentiment analysis based on the BERT-CNN-BiGRU-attention model in the context of intercultural communication barriers. *PeerJ Computer Science, 10*, e2166. https://doi.org/10.7717/peerj-cs.2166
2. Chen, N., Sun, Y., and Yan, Y. (2023). Sentiment analysis and research based on two-channel parallel hybrid neural network model with attention mechanism. *IET Control Theory & Applications, 17*, 2259-2267. https://doi.org/10.1049/cth2.12463
3. Zhao, Z., Liu, W., and Wang, K. (2023). Research on sentiment analysis method of opinion mining based on multi-model fusion transfer learning. *Journal of Big Data, 10*, 155. https://doi.org/10.1186/s40537-023-00837-x
4. Kim, J., Na, Y., Kim, K., Lee, S. R., and Chae, D.-K. (2024). SentiCSE: A Sentiment-aware Contrastive Sentence Embedding Framework with Sentiment-guided Textual Similarity. *Proceedings of LREC-COLING 2024*, 14693-14704. https://aclanthology.org/2024.lrec-main.1280/
