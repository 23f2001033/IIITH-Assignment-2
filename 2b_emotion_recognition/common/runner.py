"""Training and testing shared by the speech, text and fusion pipelines.

Each run trains on the split's train ids, selects the epoch with the best validation macro-F1
(early stopping), and is then evaluated once on the test ids. Run directories hold:
  model.pt, config.json, history.csv        written by train.py
  predictions.csv, metrics.json, embeddings.npz   written by test.py
"""
import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from common.data import EMOTIONS, labels, load_split, read_csv
from common.modules import FusionModel, SequenceEncoder, UnimodalModel

FEATURE_FILES = {"wavlm": "speech_wavlm.pt", "logmel": "speech_logmel.pt", "bert": "text_bert.pt"}
_CACHE = {}


def load_features(cache_dir, kind, device="cpu", speaker_norm=False, data_dir=None):
    """Feature dictionaries are loaded once per process; on a GPU they are kept there as float16
    (WavLM features for all 2800 clips are about 4 GB), so padding happens on the GPU.

    speaker_norm: subtract each speaker's mean and divide by their standard deviation, per feature
    (and per layer), computed over all of that speaker's frames. It needs no labels, only knowing
    which clips come from the same speaker, and removes voice-specific offsets that do not carry emotion.
    """
    key = (cache_dir, kind, str(device), speaker_norm)
    if key not in _CACHE:
        if speaker_norm:
            base = load_features(cache_dir, kind, device)
            speaker = {r["id"]: r["speaker"] for r in read_csv(os.path.join(data_dir, "manifest.csv"))}
            stats = {}
            for spk in set(speaker.values()):
                # running sums keep GPU memory flat (all WavLM frames of one speaker would be ~4.5 GB)
                total, total_sq, n = 0.0, 0.0, 0
                for i, v in base.items():
                    if speaker[i] == spk:
                        x = v.float()
                        total = total + x.sum(dim=-2, keepdim=True)
                        total_sq = total_sq + (x * x).sum(dim=-2, keepdim=True)
                        n += x.shape[-2]
                mean = total / n
                stats[spk] = (mean, (total_sq / n - mean * mean).clamp_min(0).sqrt() + 1e-5)
            _CACHE[key] = {i: ((v.float() - stats[speaker[i]][0]) / stats[speaker[i]][1]).half() for i, v in base.items()}
            _CACHE.pop((cache_dir, kind, str(device), False), None)  # free the raw copy; reloaded if needed
        else:
            features = torch.load(os.path.join(cache_dir, FEATURE_FILES[kind]))
            _CACHE[key] = {k: v.to(device) for k, v in features.items()} if str(device) != "cpu" else features
    return _CACHE[key]


def make_encoder(kind):
    if kind == "wavlm":
        return SequenceEncoder(768, n_layers=13, lstm_layers=2)
    if kind == "logmel":
        return SequenceEncoder(80, n_layers=None, lstm_layers=2)
    if kind == "bert":
        return SequenceEncoder(768, n_layers=13, lstm_layers=1)
    raise ValueError(kind)


def build_model(config):
    if config["pipeline"] == "fusion":
        return FusionModel(make_encoder(config["speech_features"]), make_encoder("bert"), mode=config["fusion"])
    return UnimodalModel(make_encoder(config["features"]))


def _pad(seqs):
    """Pad [n_layers?, T, D] tensors along T; returns float tensor and boolean mask [B, T]."""
    lengths = [s.shape[-2] for s in seqs]
    t_max = max(lengths)
    shape = (len(seqs),) + tuple(seqs[0].shape[:-2]) + (t_max, seqs[0].shape[-1])
    out = torch.zeros(shape, dtype=torch.float32, device=seqs[0].device)
    mask = torch.zeros(len(seqs), t_max, dtype=torch.bool, device=seqs[0].device)
    for i, (s, n) in enumerate(zip(seqs, lengths)):
        out[i, ..., :n, :] = s.float()
        mask[i, :n] = True
    return out, mask


class Batches:
    """Minimal batch iterator over ids; features are looked up in the cached dictionaries."""

    def __init__(self, ids, y, config, cache_dir, batch_size=32, shuffle=False, seed=0, device="cpu", data_dir=None):
        self.ids, self.y, self.config, self.batch_size, self.shuffle = list(ids), y, config, batch_size, shuffle
        self.rng = random.Random(seed)
        norm = config.get("speaker_norm", False)
        if config["pipeline"] == "fusion":
            self.speech = load_features(cache_dir, config["speech_features"], device, norm, data_dir)
            self.text = load_features(cache_dir, "bert", device)
        else:
            self.x = load_features(cache_dir, config["features"], device,
                                   norm and config["pipeline"] == "speech", data_dir)

    def __iter__(self):
        ids = list(self.ids)
        if self.shuffle:
            self.rng.shuffle(ids)
        for start in range(0, len(ids), self.batch_size):
            chunk = ids[start:start + self.batch_size]
            batch = {"ids": chunk, "y": torch.tensor([self.y[i] for i in chunk])}
            if self.config["pipeline"] == "fusion":
                batch["x_speech"], batch["mask_speech"] = _pad([self.speech[i] for i in chunk])
                batch["x_text"], batch["mask_text"] = _pad([self.text[i] for i in chunk])
            else:
                batch["x"], batch["mask"] = _pad([self.x[i] for i in chunk])
            yield batch


def _to(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def predict(model, batches, device):
    model.eval()
    out = {"ids": [], "y": [], "logits": [], "embedding": [], "speech_embedding": [], "text_embedding": [], "gate": []}
    for batch in batches:
        result = model(_to(batch, device))
        out["ids"] += batch["ids"]
        out["y"].append(batch["y"])
        for key in ("logits", "embedding", "speech_embedding", "text_embedding", "gate"):
            if key in result:
                out[key].append(result[key].float().cpu())
    return {k: (torch.cat(v).numpy() if v and torch.is_tensor(v[0]) else v) for k, v in out.items()}


def scores(y_true, logits):
    pred = logits.argmax(axis=1)
    return {"accuracy": float(accuracy_score(y_true, pred)),
            "macro_f1": float(f1_score(y_true, pred, average="macro", labels=list(range(len(EMOTIONS))), zero_division=0))}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(config, data_dir, cache_dir, out_dir, device):
    set_seed(config["seed"])
    os.makedirs(out_dir, exist_ok=True)
    split, y = load_split(data_dir, config["split"]), labels(data_dir)
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()
    train_batches = Batches(split["train"], y, config, cache_dir, config["batch_size"], shuffle=True,
                            seed=config["seed"], device=device, data_dir=data_dir)
    val_batches = Batches(split["val"], y, config, cache_dir, 64, device=device, data_dir=data_dir)
    best, best_state, history, waited = None, None, [], 0
    start = time.time()
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total, n = 0.0, 0
        for batch in train_batches:
            batch = _to(batch, device)
            loss = loss_fn(model(batch)["logits"], batch["y"])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total, n = total + loss.item() * len(batch["ids"]), n + len(batch["ids"])
        val = predict(model, val_batches, device)
        val_loss = float(loss_fn(torch.from_numpy(val["logits"]), torch.from_numpy(val["y"])).item())
        val_scores = scores(val["y"], val["logits"])
        history.append({"epoch": epoch, "train_loss": round(total / n, 4), "val_loss": round(val_loss, 4),
                        "val_accuracy": round(val_scores["accuracy"], 4), "val_macro_f1": round(val_scores["macro_f1"], 4)})
        key = (val_scores["macro_f1"], -val_loss)
        if best is None or key > best:
            best, waited = key, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
        else:
            waited += 1
            if waited >= config["patience"]:
                break
    torch.save(best_state, os.path.join(out_dir, "model.pt"))
    config = dict(config, best_epoch=best_epoch, best_val_macro_f1=best[0], train_seconds=round(time.time() - start, 1))
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=1)
    with open(os.path.join(out_dir, "history.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    return config


def test(run_dir, data_dir, cache_dir, device):
    with open(os.path.join(run_dir, "config.json")) as f:
        config = json.load(f)
    split, y = load_split(data_dir, config["split"]), labels(data_dir)
    model = build_model(config).to(device)
    model.load_state_dict(torch.load(os.path.join(run_dir, "model.pt"), map_location=device))
    out = predict(model, Batches(split["test"], y, config, cache_dir, 64, device=device, data_dir=data_dir), device)
    probs = torch.softmax(torch.from_numpy(out["logits"]), dim=1).numpy()
    pred = probs.argmax(axis=1)
    metrics = dict(scores(out["y"], out["logits"]), n_test=len(out["ids"]),
                   per_class_f1=dict(zip(EMOTIONS, f1_score(out["y"], pred, average=None, labels=list(range(7)), zero_division=0).round(4).tolist())),
                   confusion=confusion_matrix(out["y"], pred, labels=list(range(7))).tolist())
    for name, module in (("layer_weights", getattr(getattr(model, "encoder", None), "layer_weights", None)),
                         ("speech_layer_weights", getattr(getattr(model, "speech", None), "layer_weights", None)),
                         ("text_layer_weights", getattr(getattr(model, "text", None), "layer_weights", None))):
        if module is not None:
            metrics[name] = torch.softmax(module.logits.detach().cpu(), 0).numpy().round(4).tolist()
    if len(out["gate"]):
        metrics["mean_gate_speech"] = float(out["gate"][:, 0].mean())
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(dict(metrics, **{k: config[k] for k in ("pipeline", "split", "seed")}), f, indent=1)
    with open(os.path.join(run_dir, "predictions.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "true", "pred", "confidence"] + ["p_" + e for e in EMOTIONS])
        for i, rid in enumerate(out["ids"]):
            writer.writerow([rid, EMOTIONS[out["y"][i]], EMOTIONS[pred[i]], round(float(probs[i].max()), 4)]
                            + [round(float(p), 4) for p in probs[i]])
    np.savez_compressed(os.path.join(run_dir, "embeddings.npz"), ids=np.array(out["ids"]), y=out["y"],
                        **{k: out[k] for k in ("embedding", "speech_embedding", "text_embedding", "gate") if len(out[k])})
    return metrics


def cli(pipeline, stage):
    parser = argparse.ArgumentParser(description="{} pipeline: {}".format(pipeline, stage))
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--cache_dir", default="cache")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    if stage == "train":
        parser.add_argument("--split", required=True)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--epochs", type=int, default=40)
        parser.add_argument("--patience", type=int, default=8)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--weight_decay", type=float, default=1e-2)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--speaker_norm", action="store_true", help="per-speaker feature normalisation (speech)")
        if pipeline == "speech":
            parser.add_argument("--features", choices=["wavlm", "logmel"], default="wavlm")
        if pipeline == "fusion":
            parser.add_argument("--fusion", choices=["gated", "concat"], default="gated")
            parser.add_argument("--speech_features", choices=["wavlm", "logmel"], default="wavlm")
    args = parser.parse_args()
    if stage == "train":
        config = {k: v for k, v in vars(args).items() if k not in ("data_dir", "cache_dir", "run_dir", "device")}
        config["pipeline"] = pipeline
        if pipeline == "text":
            config["features"] = "bert"
        result = train(config, args.data_dir, args.cache_dir, args.run_dir, args.device)
        print(json.dumps({k: result[k] for k in ("split", "seed", "best_epoch", "best_val_macro_f1", "train_seconds")}))
    else:
        metrics = test(args.run_dir, args.data_dir, args.cache_dir, args.device)
        print(json.dumps({k: metrics[k] for k in ("accuracy", "macro_f1", "n_test")}))
