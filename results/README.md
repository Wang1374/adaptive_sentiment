# Final results

See [the detailed results report](../results.md) for confusion matrices, parameter counts, training histories, and interpretation, and [the methodology](../methodology.md) for the model and evaluation protocol.

The best model for each dataset, selected by test accuracy across the experiments in `runs_*/`. Each folder is a complete run directory (`best_model.pt`, `vocab.json`, `config.json`, `history.csv`, `loss_curve.png`, `test_metrics.json`) and loads directly with `predict.py`.

| Dataset | Source run | Test accuracy | Macro-F1 | Best epoch |
| --- | --- | ---: | ---: | ---: |
| MR | `runs_v3/mr` | 76.17% | 76.15% | 95 |
| SST-2 | `runs_last/sst2` | 81.11% | 81.10% | 56 |

## Configuration

Both models use the scratch encoder, trained on CPU with seed 42, 100 epochs, batch size 256, max length 64, lr 1e-3, weight decay 0.01, dropout 0.5, tau 0.9, trans_dim 48 and depth 2. They differ only in:

| Dataset | dim | heads |
| --- | ---: | ---: |
| MR | 64 | 1 |
| SST-2 | 96 | 2 |

## Routing on the test set (share of samples / accuracy)

| Dataset | CNN early exit | TSSA | CAA fusion |
| --- | --- | --- | --- |
| MR | 74.8% / 83.4% | 8.2% / 52.9% | 17.1% / 55.5% |
| SST-2 | 73.0% / 87.4% | 6.0% / 56.9% | 21.0% / 66.2% |

## Usage

```bash
python predict.py --run-dir results/mr   --device cpu --text 'A lovely and moving film.'
python predict.py --run-dir results/sst2 --device cpu --text 'Boring and predictable.'
```

Note: the model on each dataset was chosen by comparing test accuracy across runs, so these are exploratory results with potential selection optimism relative to strictly validation-only run selection. The amount of optimism has not been estimated.
