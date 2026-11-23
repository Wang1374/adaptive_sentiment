# Confidence-Aware Adaptive Sentiment Analysis

A PyTorch research implementation of binary English sentiment classification on **MR** and **SST-2**. The model combines a small shared Transformer encoder, TextCNN, a reduced-dimension TSSA branch, and one-dimensional Context Anchor Attention (CAA). Confidence-based routing lets individual examples exit through the CNN or continue to TSSA and, when required by the routing rule, feature fusion.

The research question is predictive: **Can a compact classifier allocate additional feature-extraction computation according to prediction confidence while retaining useful sentiment classification performance?** The current experiments establish classification results, parameter counts, and route usage. Runtime improvements and the accuracy benefit of routing remain to be measured against controlled baselines.

## Documentation

| Document | Contents |
| --- | --- |
| [Methodology](methodology.md) | Research design, data processing, model equations, routing, training objective, and validation protocol |
| [Results](results.md) | Selected experiments, configurations, confusion matrices, route statistics, parameter counts, and limitations |
| [Saved model directories](results/README.md) | Checkpoint contents and inference commands |

## Selected experimental results

These are the final runs selected for the current research report. Both use a randomly initialized **scratch encoder**, not pretrained BERT or DistilBERT.

| Dataset | Source run | Test accuracy | Macro-F1 | Best epoch | Model parameters |
| --- | --- | ---: | ---: | ---: | ---: |
| MR | `runs_v3/mr` | 76.17% | 76.15% | 95 | 1,252,308 |
| SST-2 | `runs_last/sst2` | 81.11% | 81.10% | 56 | 1,617,318 |

Complete copies of these runs are included in `results/mr/` and `results/sst2/`. Within each run, the checkpoint is selected by validation accuracy, with validation loss breaking ties. The final run for each dataset was selected after comparing test accuracy across experiments. Consequently, these are exploratory, test-informed results, not an untouched final test assessment. The magnitude of selection optimism has not been estimated.

## Actual inference flow

```text
Text -> token IDs and padding mask -> shared Transformer encoder
                                      |
                                      +-> TextCNN -> confidence C_cnn
                                      |               |
                                      |               +-> C_cnn >= tau: CNN output
                                      |               |
                                      +-> TSSA <-------+-- C_cnn < tau
                                            |
                                            +-> C_tssa >= C_cnn: TSSA output
                                            |
                                            +-> C_tssa < C_cnn: CAA(CNN, TSSA) output
```

The CNN and TSSA branches receive the **same shared encoder output**; TSSA does not take CNN features as its input. The decisions are sequential, while the feature representations come from separate branches. The default and selected-run threshold is `tau=0.9`. Only the first decision uses this absolute threshold; the second compares the two branches' confidence scores.

All examples execute the shared Transformer encoder and CNN. Early exit skips the subsequent TSSA and CAA work. Routing uses softmax confidence, not sentence length or a directly measured semantic-complexity score. Training and validation compute all branches; sparse execution is used for test evaluation and prediction.

## Repository layout

```text
data.py                     CSV loading, tokenization, training-only vocabulary, padding
model.py                    Shared encoder, CNN, TSSA, CAA, routing, training loss
train.py                    Training, validation checkpoint selection, test evaluation
predict.py                  Load a saved run and predict sentiment
requirements.txt            Core dependency constraints
requirements-pretrained.txt Optional pretrained-encoder dependencies
methodology.md              Detailed research method
results.md                  Detailed experimental evidence
results/mr/                 Complete selected MR run
results/sst2/               Complete selected SST-2 run
runs*/                      Earlier experiment records; most weights are ignored by Git
dataset/                    Local data; excluded from Git
```

## Installation and environment

Use Python 3.10 or later for the current project setup:

```bash
python -m pip install -r requirements.txt
```

The dependency constraints are `torch>=2.6,<3`, `matplotlib>=3.8,<4`, and `tqdm>=4.66,<5`. The saved experiments record **PyTorch 2.2.2**, CPU execution, and eight CPU threads. They do not record the exact Python version, CPU model, operating system, IDE version, or the other package versions. Installing the current requirements therefore does not recreate the original environment exactly; a complete historical lockfile is unavailable.

For the optional pretrained path:

```bash
python -m pip install -r requirements-pretrained.txt
```

This adds `transformers>=4.48,<5`. The optional path is implemented but is not the source of the selected results.

## Data preparation

Provide the existing local classification splits as follows:

```text
dataset/
  mr/prepared_seed42/{train,validation,test}.csv
  sst2/prepared_seed42/{train,validation,test}.csv
```

The braces above abbreviate three separate CSV files. Each file must have `text` and `label` columns, nonempty text, and labels `0` (negative) or `1` (positive).

| Dataset | Training | Validation | Test | Saved vocabulary size |
| --- | ---: | ---: | ---: | ---: |
| MR | 8,530 | 1,066 | 1,066 | 17,089 |
| SST-2 | 6,909 | 872 | 1,821 | 13,846 |

The loader preserves supplied splits and performs no repartitioning. These are the locally prepared splits used in the experiments. In particular, the SST-2 training size differs from the 67,349 training examples reported in the SentiCSE paper; do not present the local split as an identical benchmark protocol.

The repository does not download or distribute the raw data, and does not contain a script that reconstructs these exact split files from an upstream release. Supply the same data to reproduce training. `data.py` contains a legacy Windows data-root default; always pass your own `--data-root`.

## Predict with the selected checkpoints

After installing dependencies, prediction needs no raw training data:

```bash
python predict.py --run-dir results/mr --device cpu --text 'A lovely and moving film.' --text 'Boring and predictable.'
python predict.py --run-dir results/sst2 --device cpu --text 'A thoughtful and entertaining story.'
```

Output is JSON containing `text`, `label`, `confidence`, and `route`. Route names are `cnn`, `transformer` (the TSSA branch), and `fusion`. Confidence is the maximum two-class softmax probability. Prediction uses the threshold in the saved configuration; `--tau` can override it, but changing the threshold changes the evaluation setting.

## Train the selected configurations

The following commands reproduce the saved argument settings using fresh output directories. They do not guarantee bitwise-identical results across software versions or hardware.

```bash
python train.py --data-root dataset --dataset mr --encoder scratch --device cpu --cpu-threads 8 --seed 42 --epochs 100 --batch-size 256 --max-length 64 --max-vocab 20000 --min-freq 1 --dim 64 --trans-dim 48 --heads 1 --depth 2 --dropout 0.5 --lr 0.001 --weight-decay 0.01 --tau 0.9 --aux-weight 0.4 --calibration-weight 0.1 --output-dir runs_reproduce_mr

python train.py --data-root dataset --dataset sst2 --encoder scratch --device cpu --cpu-threads 8 --seed 42 --epochs 100 --batch-size 256 --max-length 64 --max-vocab 20000 --min-freq 1 --dim 96 --trans-dim 48 --heads 2 --depth 2 --dropout 0.5 --lr 0.001 --weight-decay 0.01 --tau 0.9 --aux-weight 0.4 --calibration-weight 0.1 --output-dir runs_reproduce_sst2
```

Use a new output directory for each experiment. Training refuses to overwrite a nonempty dataset run directory. The training default device is `cuda`; explicitly select `cpu` or `auto` when CUDA is unavailable. `auto` selects CUDA when available and otherwise CPU. Both datasets can be trained with `--dataset all`, which applies the same arguments to each; the selected experiments above use different dimensions and head counts, so separate commands are needed.

For a short functional run, use fewer epochs and a fresh output directory. Its scores are not the selected research results.

## Optional pretrained encoder

```bash
python train.py --data-root dataset --dataset mr --encoder pretrained --model-name-or-path /path/to/local/distilbert-base-uncased --device auto --epochs 8 --batch-size 32 --output-dir runs_pretrained
```

The directory must contain the model weights, configuration, and tokenizer. To permit a download, use `--model-name-or-path distilbert/distilbert-base-uncased --allow-download`. The pretrained encoder uses `--encoder-lr` (default `2e-5`); other layers use `--lr`. A `model_name_or_path` field also appears in scratch-run configurations, but is unused when `encoder` is `scratch`.

## Model and training summary

- The shared encoder contains word and position embeddings and one standard bidirectional Transformer encoder layer.
- TextCNN applies parallel token-axis convolutions with kernel sizes 3, 5, and 7, followed by projection, a residual connection, and layer normalization.
- The TSSA branch projects to dimension 48, applies two TSSA blocks, and projects back. TSSA uses feature second moments without constructing a token-by-token attention matrix. The shared encoder still performs conventional self-attention.
- CAA computes a sigmoid gate for each token and feature channel to mix CNN and TSSA features. This gate is distinct from the softmax used for sentiment confidence.
- Each branch uses the same classifier on concatenated masked mean and max pooling.
- Training uses routed cross-entropy plus `0.4 * mean branch cross-entropy + 0.1 * mean branch Brier loss`. AdamW and gradient clipping at norm 1.0 are used.

The MR model has one TSSA head, so its softmax across heads is identically one at valid positions. The remaining second-moment computation still operates, but the implementation does not learn a nontrivial distribution across heads in that configuration. See [methodology.md](methodology.md) for the mathematical details.

## Saved artifacts

| File | Meaning |
| --- | --- |
| `best_model.pt` | Best validation checkpoint, selected by accuracy then lower loss |
| `last_model.pt` | Final epoch weights; written during training, generally excluded from Git |
| `config.json` | Arguments, architecture, split sizes, class counts, and recorded environment information |
| `history.csv` | Per-epoch training/validation loss, accuracy, and route fractions |
| `loss_curve.png` | Training and validation total-loss curves |
| `test_metrics.json` | Test accuracy, Macro-F1, confusion matrix, route statistics, and best-epoch information |
| `vocab.json` | Scratch vocabulary fitted on training text only |
| `tokenizer/`, `encoder_config/` | Additional artifacts for pretrained runs |

## Interpretation and remaining experiments

The saved results support reporting a compact classifier and approximately 73-75% CNN early exits on these test splits. They do not establish a particular runtime speedup, a training-time reduction, or superiority over an always-compute baseline. Routing changes executed computation, not the model's stored parameter count.

The current evidence comes from one seed per selected configuration. Cross-validation, multi-seed confidence intervals, systematic hyperparameter search, threshold sweeps, controlled branch ablations, and matched runtime benchmarks have not been completed in this repository. [results.md](results.md) explains the observed generalization gaps, confidence limitations, and test-informed run selection.

## Research context

The current study investigates selective computation after a compact shared encoder. It draws on hybrid sentiment-classification literature and uses locally prepared sentiment splits associated with the SentiCSE reference workflow. It does not implement SentiCSE's contrastive pretraining or claim to reproduce the reference papers' published scores.

- Bi and Zhang (2024): [BERT-CNN-BiGRU-attention sentiment model](https://doi.org/10.7717/peerj-cs.2166).
- Chen, Sun, and Yan (2023): [BERT-based parallel CNN-BiLSTM-attention model](https://doi.org/10.1049/cth2.12463).
- Zhao, Liu, and Wang (2023): [Multi-model fusion transfer learning for sentiment analysis](https://doi.org/10.1186/s40537-023-00837-x).
- Kim et al. (2024): [SentiCSE](https://aclanthology.org/2024.lrec-main.1280/), with [reference code](https://github.com/nayohan/SentiCSE).

The TSSA code comments identify module 129 and the [ToST implementation](https://github.com/RobinWu218/ToST) as its reference. The CAA code comments identify an adaptation of module 083; complete bibliographic provenance for that module remains to be documented.
