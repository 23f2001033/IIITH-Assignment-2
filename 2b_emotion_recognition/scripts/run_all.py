"""Train and test every model variant on every split, in one process so the feature cache loads once.

Variants: speech (WavLM), speech (log-mel baseline), text (BERT), fusion (gated), fusion (concat).
Splits: random_s0-2, word_s0-2, speaker_OAF2YAF, speaker_YAF2OAF. Each run's seed is the split's seed.
Finished runs (metrics.json present) are skipped, so the script can be resumed.

usage: python scripts/run_all.py --data_dir data --cache_dir cache --runs_dir runs [--quick]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from common.runner import test, train  # noqa: E402

VARIANTS = {
    "speech_wavlm": {"pipeline": "speech", "features": "wavlm"},
    "speech_logmel": {"pipeline": "speech", "features": "logmel"},
    "text_bert": {"pipeline": "text", "features": "bert"},
    "fusion_gated": {"pipeline": "fusion", "fusion": "gated", "speech_features": "wavlm"},
    "fusion_concat": {"pipeline": "fusion", "fusion": "concat", "speech_features": "wavlm"},
}
# Speaker-independent results vary strongly between training runs, so that split family is repeated
# with five seeds, with and without per-speaker normalisation (plan "speaker_seeds").
SPEAKER_VARIANTS = {
    "speech_wavlm": VARIANTS["speech_wavlm"],
    "speech_logmel": VARIANTS["speech_logmel"],
    "fusion_gated": VARIANTS["fusion_gated"],
    "fusion_concat": VARIANTS["fusion_concat"],
    "speech_wavlm_spknorm": dict(VARIANTS["speech_wavlm"], speaker_norm=True),
    "fusion_gated_spknorm": dict(VARIANTS["fusion_gated"], speaker_norm=True),
}
SPLITS = ["random_s0", "random_s1", "random_s2", "word_s0", "word_s1", "word_s2", "speaker_OAF2YAF", "speaker_YAF2OAF"]
DEFAULTS = {"epochs": 40, "patience": 8, "lr": 1e-3, "weight_decay": 1e-2, "batch_size": 32}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--cache_dir", default="cache")
    parser.add_argument("--runs_dir", default="runs")
    parser.add_argument("--quick", action="store_true", help="one split, 2 epochs: checks the setup")
    parser.add_argument("--plan", choices=["main", "speaker_seeds"], default="main")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.plan == "main":
        splits = SPLITS[:1] + SPLITS[3:4] + SPLITS[6:7] if args.quick else SPLITS
        jobs = [(split, int(split[-1]) if split[-2] == "s" else 0, name, variant, "{}__{}".format(name, split))
                for split in splits for name, variant in VARIANTS.items()]
    else:
        splits = ["speaker_OAF2YAF", "speaker_YAF2OAF"]
        jobs = [(split, seed, name, variant, "{}__{}__seed{}".format(name, split, seed))
                for seed in range(args.seeds) for split in splits for name, variant in SPEAKER_VARIANTS.items()]
        # normalised variants last: building their features frees the raw WavLM copy on the GPU
        jobs.sort(key=lambda job: job[3].get("speaker_norm", False))
    start = time.time()
    for split, seed, name, variant, run_name in jobs:
        run_dir = os.path.join(args.runs_dir, run_name)
        if os.path.exists(os.path.join(run_dir, "metrics.json")):
            continue
        config = dict(DEFAULTS, **variant, split=split, seed=seed, variant=name)
        if args.quick:
            config["epochs"] = 2
        result = train(config, args.data_dir, args.cache_dir, run_dir, args.device)
        metrics = test(run_dir, args.data_dir, args.cache_dir, args.device)
        print("{:6.0f}s {:22s} {:16s} seed {} epoch {:2d}  val F1 {:.3f}  test acc {:.3f}  F1 {:.3f}".format(
            time.time() - start, name, split, seed, result["best_epoch"], result["best_val_macro_f1"],
            metrics["accuracy"], metrics["macro_f1"]), flush=True)
    with open(os.path.join(args.runs_dir, "done_{}.json".format(args.plan)), "w") as f:
        json.dump({"seconds": round(time.time() - start), "splits": splits,
                   "variants": sorted({job[2] for job in jobs})}, f)


if __name__ == "__main__":
    main()
