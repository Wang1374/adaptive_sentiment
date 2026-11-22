"""Load a trained run and predict sentiment for English texts, showing the route each one took."""

import argparse
import json
from pathlib import Path

import torch

from data import BatchCollator
from model import ROUTES, AdaptiveSentiment
from train import choose_device


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--text", required=True, action="append")
    parser.add_argument("--tau", type=float, default=None, help="Defaults to the training threshold")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    run_dir = args.run_dir
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    tau = config["tau"] if args.tau is None else args.tau
    if not 0.5 <= tau <= 1.0:
        parser.error("tau must be in [0.5, 1.0]")
    device = choose_device(args.device)
    pretrained = None
    if config["encoder"] == "pretrained":
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(run_dir / "tokenizer", local_files_only=True)
        encoder_config = AutoConfig.from_pretrained(run_dir / "encoder_config", local_files_only=True)
        pretrained = AutoModel.from_config(encoder_config)
        collator = BatchCollator(config["max_length"], tokenizer=tokenizer)
    else:
        vocab = json.loads((run_dir / "vocab.json").read_text(encoding="utf-8"))
        collator = BatchCollator(config["max_length"], vocab=vocab)
    model = AdaptiveSentiment(**config["model_args"], vocab_size=config["vocab_size"], pretrained=pretrained)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=True))
    model.to(device).eval()
    ids, mask = collator.encode(args.text)
    logits, routes = model.predict(ids.to(device), mask.to(device), tau)
    scores, labels = logits.float().softmax(-1).max(-1)
    result = [{"text": text, "label": ("negative", "positive")[label],
               "confidence": score, "route": ROUTES[route]}
              for text, label, score, route in zip(args.text, labels.tolist(), scores.tolist(), routes.tolist())]
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
