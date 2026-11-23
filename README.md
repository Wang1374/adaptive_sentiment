# Confidence-Aware Adaptive Sentiment Analysis (MR / SST-2)

A PyTorch implementation of binary English sentiment classification with TextCNN, TSSA, and one-dimensional Context Anchor Attention (CAA). Each example follows a confidence-based inference route: exit through the CNN, use the deeper TSSA branch, or fuse both representations with CAA.

Labels are `0 = negative` and `1 = positive`.

## Data and evaluation protocol

The project reads the existing SentiCSE classification splits without repartitioning them:

```text
dataset/
  mr/prepared_seed42/
    train.csv
    validation.csv
    test.csv
  sst2/prepared_seed42/
    train.csv
    validation.csv
    test.csv
```

Every CSV must contain `text` and `label` columns. Supply the dataset root with `--data-root`; raw datasets are excluded from Git.

| Dataset | Training | Validation | Test |
| --- | ---: | ---: | ---: |
| MR | 8,530 | 1,066 | 1,066 |
| SST-2 | 6,909 | 872 | 1,821 |

This is directly supervised text classification. It retains the reference project's local splits but does not use SentiCSE quadruplet contrastive learning, sentiment MLM, or Spearman-based checkpoint selection. MR uses a local stratified split, and SST-2 uses a locally prepared version. Compare these results with experiments that use the same splits.

## Installation

Use Python 3.10 or later:

```bash
python -m pip install -r requirements.txt
```

For the optional pretrained encoder, also install:

```bash
python -m pip install -r requirements-pretrained.txt
```

## Predict with the included models

The `results/` directory contains complete scratch-encoder runs, including model weights and vocabularies:

```bash
python predict.py --run-dir results/mr --device cpu --text 'A lovely and moving film.' --text 'Boring and predictable.'
python predict.py --run-dir results/sst2 --device cpu --text 'A thoughtful and entertaining story.'
```

Predictions are printed as JSON with the input text, sentiment label, softmax confidence, and route (`cnn`, `transformer`, or `fusion`). Prediction uses the run's saved threshold unless overridden with `--tau`.

| Dataset | Source run | Test accuracy | Macro-F1 | Best epoch |
| --- | --- | ---: | ---: | ---: |
| MR | `runs_v3/mr` | 76.17% | 76.15% | 95 |
| SST-2 | `runs_last/sst2` | 81.11% | 81.10% | 56 |

Within each run, checkpoints were selected by validation accuracy, with validation loss as the tie-breaker. The final models in `results/` were then chosen by comparing test accuracy across runs, so these reported results are optimistic relative to strictly validation-only model selection. See [the final results documentation](results/README.md) for configurations and route statistics.

## Train a scratch encoder

The scratch encoder is a small, randomly initialized, bidirectional BERT-style Transformer. It runs offline, uses no pretrained weights, and builds its vocabulary from the training split only.

```bash
python train.py --data-root dataset --dataset all --encoder scratch --device auto --epochs 8 --output-dir runs_offline
```

Use `--dataset mr` or `--dataset sst2` to train one dataset. `--device auto` selects CUDA when available and otherwise uses CPU. The training command defaults to `--device cuda`, so explicitly use `--device cpu` or `--device auto` on machines without CUDA.

The data-root default in `data.py` is a path from the original Windows environment. Pass `--data-root` to point to your own data. On Windows, use `python` or the path to your Python executable and adjust file paths as needed.

Training defaults include 100 epochs, batch size 256, maximum sequence length 64, feature dimension 64, TSSA dimension 48, one head, two TSSA blocks, dropout 0.5, learning rate `1e-3`, weight decay `0.01`, seed 42, and routing threshold `tau=0.9`. The short example above overrides the epoch count for convenience.

## Train a pretrained encoder

To fine-tune a local lightweight BERT-family model, supply a directory containing its weights, configuration, and tokenizer:

```bash
python train.py --data-root dataset --dataset all --encoder pretrained --model-name-or-path /path/to/distilbert-base-uncased --device auto --epochs 8 --batch-size 32 --output-dir runs_pretrained
```

Alternatively, allow a download with `--model-name-or-path distilbert/distilbert-base-uncased --allow-download`. The pretrained encoder uses `--encoder-lr` (default `2e-5`); the remaining layers use `--lr`.

## Architecture and routing

| File | Responsibility |
| --- | --- |
| `data.py` | Read CSV splits, tokenize text, build the scratch vocabulary, and dynamically pad batches |
| `model.py` | Implement the encoders, TextCNN, TSSA, 1D CAA, confidence routing, and training loss |
| `train.py` | Train, select validation checkpoints, evaluate the test split, and save curves and metrics |
| `predict.py` | Restore a saved run and output labels, confidence scores, and routes |

The encoder first produces token representations with shape `[B, L, D]`.

- **TextCNN:** One-dimensional convolutions with kernel sizes 3, 5, and 7 slide along the token axis to capture local patterns at different spans.
- **TSSA:** The deeper branch projects `D` to `trans_dim`, applies TSSA blocks with padding masks, residual connections, and feed-forward layers, then projects back to `D`. The operator follows module 129, using softmax across heads and second-moment estimates over valid tokens. The source references the [official ToST implementation](https://github.com/RobinWu218/ToST).
- **1D CAA:** Adapted from module 083, this branch uses local averaging and sequential depthwise convolutions to produce a sigmoid gate for each token and channel, fusing the CNN and TSSA representations.

All three branches share one classification head. Pooling, statistics, and convolution boundaries account for padding.

Confidence is the maximum **softmax** probability over the two sentiment logits. Inference routes each example independently:

1. If `C_cnn >= tau`, return the CNN prediction without executing TSSA or CAA for that example.
2. Otherwise, execute TSSA. If `C_trans >= C_cnn`, return the TSSA prediction.
3. For the remaining examples, execute 1D CAA and return the fused prediction.

The default threshold is `tau=0.9`. A larger threshold generally sends more examples to the deeper branch. Compare thresholds on validation data.

TSSA itself does not construct an `L x L` token-attention matrix. The preceding BERT-style encoder still has conventional self-attention costs, so end-to-end speed must be measured.

## Training objective

Training computes all branches and uses hard routing to select the main-task logits. Auxiliary cross-entropy losses supervise all three branches. A multiclass Brier loss penalizes the difference between predicted probabilities and the one-hot target:

```text
loss = main_cross_entropy
     + 0.4 * mean_branch_cross_entropy
     + 0.1 * mean_branch_brier_loss
```

The weights are configurable with `--aux-weight` and `--calibration-weight`. Brier loss is a calibration objective; it does not guarantee that confidence scores from different branches are fully calibrated. Validation uses the same loss as training, while test evaluation uses sparse inference routing.

## Saved artifacts and experiment history

Training and validation display progress, cumulative loss, and accuracy with `tqdm`. Each dataset gets its own output subdirectory. Use a fresh `--output-dir` for every experiment; training rejects a nonempty run directory. The default output root is `runs/` inside the project.

| Artifact | Contents |
| --- | --- |
| `best_model.pt` | Checkpoint with the highest validation accuracy; ties use lower validation loss |
| `last_model.pt` | Most recent epoch's weights, overwritten after every epoch |
| `config.json` | Run arguments, model configuration, split statistics, and environment details |
| `history.csv` | Per-epoch training and validation loss, accuracy, and route fractions |
| `loss_curve.png` | Training and validation total loss, updated after every epoch |
| `test_metrics.json` | Test accuracy, Macro-F1, confusion matrix, route coverage and accuracy, and checkpoint-selection details |
| `vocab.json` | Training vocabulary for the scratch encoder |
| `tokenizer/`, `encoder_config/` | Tokenizer and encoder configuration for pretrained runs |

The training loss curve accumulates values while parameters are updated. The validation curve is computed after each epoch with fixed weights and dropout disabled.

The included `runs/` directory records earlier six-epoch CPU experiments with the scratch encoder and seed 42. Other `runs_*` directories preserve experiment configurations, curves, and metrics. Their weights and vocabularies are generally excluded from Git; use the complete runs in `results/` for inference.
