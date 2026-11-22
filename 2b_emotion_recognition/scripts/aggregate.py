"""Turn the runs into the Results/ folder: accuracy tables, per-emotion scores, plots, failure cases.

Run names are <variant>__<split> (main plan) or <variant>__<split>__seed<k> (speaker splits repeated
over seeds). For speaker splits the seed-repeated runs are used whenever they exist.

usage: python scripts/aggregate.py --runs_dir runs [more_runs ...] --data_dir data --out Results
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "pleasant_surprise", "sad"]
SHORT = ["ang", "dis", "fea", "hap", "neu", "ps", "sad"]
COLOURS = ["#d62728", "#8c564b", "#9467bd", "#ff7f0e", "#7f7f7f", "#17becf", "#1f77b4"]
VARIANT_NAMES = {
    "speech_wavlm": "Speech (WavLM + BiLSTM)",
    "speech_logmel": "Speech (log-mel + BiLSTM, baseline)",
    "speech_wavlm_spknorm": "Speech (WavLM + BiLSTM) + speaker norm.",
    "text_bert": "Text (BERT + BiLSTM)",
    "fusion_gated": "Fusion (gated)",
    "fusion_concat": "Fusion (concat)",
    "fusion_gated_spknorm": "Fusion (gated) + speaker norm.",
}
FAMILIES = {"random": "Random split", "word": "Word-disjoint", "speaker": "Speaker-independent"}
SPEAKER_SPLITS = ("speaker_OAF2YAF", "speaker_YAF2OAF")


def family(split):
    return split.split("_")[0]


def load_runs(runs_dirs):
    runs = {}
    for runs_dir in runs_dirs:
        for path in glob.glob(os.path.join(runs_dir, "*", "metrics.json")):
            name = os.path.basename(os.path.dirname(path))
            parts = name.split("__")
            with open(path) as f:
                m = json.load(f)
            m.update(dir=os.path.dirname(path), variant=parts[0], split=parts[1],
                     plan="seeds" if len(parts) == 3 else "main")
            runs[name] = m
    return runs


def select(runs, variant, fam=None, split=None):
    """Runs of one variant in a split family or one split; seed-repeated runs win for speaker splits."""
    chosen = [m for m in runs.values() if m["variant"] == variant and (fam is None or family(m["split"]) == fam)
              and (split is None or m["split"] == split)]
    for spk in SPEAKER_SPLITS:
        if any(m["plan"] == "seeds" and m["split"] == spk for m in chosen):
            chosen = [m for m in chosen if not (m["split"] == spk and m["plan"] == "main")]
    return sorted(chosen, key=lambda m: (m["split"], m["seed"]))


def mean_std(values):
    if not values:
        return "-"
    if len(values) == 1:
        return "{:.1f}".format(100 * values[0])
    return "{:.1f} ± {:.1f}".format(100 * np.mean(values), 100 * np.std(values))


def write_md(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def accuracy_tables(runs, out):
    csv_rows = []
    for metric in ("accuracy", "macro_f1"):
        lines = ["| Model | " + " | ".join("{} (n)".format(v) for v in FAMILIES.values()) + " |",
                 "|---|" + "---|" * len(FAMILIES)]
        for variant, label in VARIANT_NAMES.items():
            cells = []
            for fam in FAMILIES:
                vals = [m[metric] for m in select(runs, variant, fam)]
                cells.append("{} ({})".format(mean_std(vals), len(vals)) if vals else "-")
                if vals:
                    csv_rows.append({"model": variant, "split_family": fam, "metric": metric,
                                     "mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4),
                                     "runs": len(vals), "values": ";".join("{:.4f}".format(v) for v in vals)})
            lines.append("| {} | {} |".format(label, " | ".join(cells)))
        write_md(os.path.join(out, "accuracy_table_{}.md".format(metric)), lines)
    lines = ["| Model | Train OAF → test YAF | Train YAF → test OAF |", "|---|---|---|"]
    for variant, label in VARIANT_NAMES.items():
        cells = []
        for split in SPEAKER_SPLITS:
            vals = [m["accuracy"] for m in select(runs, variant, split=split)]
            cells.append("{} [{:.1f}-{:.1f}], n={}".format(mean_std(vals), 100 * min(vals), 100 * max(vals), len(vals))
                         if len(vals) > 1 else mean_std(vals))
        lines.append("| {} | {} |".format(label, " | ".join(cells)))
    write_md(os.path.join(out, "accuracy_speaker_directions.md"), lines)
    with open(os.path.join(out, "accuracy_table.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)


def per_class(runs, out):
    rows = []
    for variant in VARIANT_NAMES:
        for fam in FAMILIES:
            f1s = [m["per_class_f1"] for m in select(runs, variant, fam)]
            if f1s:
                rows.append(dict({"model": variant, "split_family": fam, "runs": len(f1s)},
                                 **{e: round(float(np.mean([f[e] for f in f1s])), 3) for e in EMOTIONS}))
    with open(os.path.join(out, "per_emotion_f1.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, ["model", "split_family", "runs"] + EMOTIONS)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| Model | Split | " + " | ".join(EMOTIONS) + " |", "|---|---|" + "---|" * 7]
    for r in rows:
        lines.append("| {} | {} | {} |".format(VARIANT_NAMES[r["model"]], r["split_family"],
                                               " | ".join("{:.2f}".format(r[e]) for e in EMOTIONS)))
    write_md(os.path.join(out, "per_emotion_f1.md"), lines)


def confusion_plots(runs, plots):
    for variant in ("speech_wavlm", "speech_wavlm_spknorm", "text_bert", "fusion_gated"):
        fams = [fam for fam in FAMILIES if select(runs, variant, fam)]
        if not fams:
            continue
        fig, axes = plt.subplots(1, len(fams), figsize=(5 * len(fams), 4.6), squeeze=False)
        for ax, fam in zip(axes[0], fams):
            runs_f = select(runs, variant, fam)
            cm = np.sum([np.array(m["confusion"]) for m in runs_f], axis=0).astype(float)
            cm = cm / cm.sum(axis=1, keepdims=True)
            ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
            for i in range(7):
                for j in range(7):
                    ax.text(j, i, "{:.0f}".format(100 * cm[i, j]), ha="center", va="center", fontsize=7,
                            color="white" if cm[i, j] > 0.5 else "black")
            ax.set_xticks(range(7), SHORT)
            ax.set_yticks(range(7), SHORT)
            ax.set_xlabel("predicted")
            ax.set_ylabel("true")
            ax.set_title("{} ({} runs)".format(FAMILIES[fam], len(runs_f)), fontsize=10)
        fig.suptitle("{}: confusion matrix, % of each true emotion, summed over runs".format(VARIANT_NAMES[variant]))
        fig.tight_layout()
        fig.savefig(os.path.join(plots, "confusion_{}.png".format(variant)), dpi=130)
        plt.close(fig)


def tsne_plots(runs, data_dir, plots, out):
    speaker = {r["id"]: r["speaker"] for r in csv.DictReader(open(os.path.join(data_dir, "manifest.csv")))}
    rows = []
    panels = [("Temporal block (speech model)", "speech_wavlm", "embedding"),
              ("Contextual block (text model)", "text_bert", "embedding"),
              ("Fusion block (gated fusion)", "fusion_gated", "embedding"),
              ("Speech branch inside fusion", "fusion_gated", "speech_embedding"),
              ("Text branch inside fusion", "fusion_gated", "text_embedding")]
    for split in ("word_s0", "speaker_OAF2YAF"):
        fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5))
        for ax, (title, variant, key) in zip(axes, panels):
            picked = select(runs, variant, split=split)
            ax.set_xticks([])
            ax.set_yticks([])
            if not picked:
                continue
            data = np.load(os.path.join(picked[0]["dir"], "embeddings.npz"))
            if key not in data:
                continue
            x, y, ids = data[key], data["y"], data["ids"]
            spk = np.array([speaker[i] for i in ids])
            sil = silhouette_score(x, y, metric="cosine")
            sil_spk = silhouette_score(x, spk, metric="cosine") if len(set(spk)) > 1 else None
            rows.append({"split": split, "block": title, "run": os.path.basename(picked[0]["dir"]),
                         "silhouette_emotion": round(float(sil), 3),
                         "silhouette_speaker": round(float(sil_spk), 3) if sil_spk is not None else ""})
            emb = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(x)
            for c in range(7):
                for s, marker in (("OAF", "o"), ("YAF", "^")):
                    sel = (y == c) & (spk == s)
                    ax.scatter(emb[sel, 0], emb[sel, 1], s=9, c=COLOURS[c], marker=marker, alpha=0.8,
                               label=EMOTIONS[c] if s == spk[0] else None)
            ax.set_title("{}\nsilhouette (emotion) = {:.2f}".format(title, sil), fontsize=10)
        axes[0].legend(fontsize=7, markerscale=2, loc="best")
        fig.suptitle("t-SNE of test-set representations, split {} (circle = OAF, triangle = YAF)".format(split))
        fig.tight_layout()
        fig.savefig(os.path.join(plots, "tsne_{}.png".format(split)), dpi=120)
        plt.close(fig)
    with open(os.path.join(out, "cluster_separability.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, ["split", "block", "run", "silhouette_emotion", "silhouette_speaker"])
        writer.writeheader()
        writer.writerows(rows)


def gate_and_layer_plot(runs, plots):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for fam, colour in zip(FAMILIES, ("#1f77b4", "#2ca02c", "#d62728")):
        gates = [np.load(os.path.join(m["dir"], "embeddings.npz"))["gate"][:, 0] for m in select(runs, "fusion_gated", fam)]
        if gates:
            g = np.concatenate(gates)
            axes[0].hist(g, bins=np.linspace(0, 1, 41), histtype="step", lw=1.8, color=colour,
                         label="{} (mean {:.3f})".format(FAMILIES[fam], g.mean()))
    axes[0].set_xlabel("weight on speech (1 - weight on text), per test clip")
    axes[0].set_ylabel("test clips")
    axes[0].set_title("Gated fusion: reliance on speech vs text", fontsize=10)
    axes[0].legend(fontsize=8)
    for variant, colour in (("speech_wavlm", "#1f77b4"), ("text_bert", "#ff7f0e")):
        weights = [m["layer_weights"] for m in runs.values() if m["variant"] == variant and "layer_weights" in m]
        if weights:
            axes[1].plot(range(13), np.mean(weights, axis=0), "-o", color=colour, ms=3,
                         label="{} ({} runs)".format("WavLM layers, speech model" if variant == "speech_wavlm"
                                                     else "BERT layers, text model", len(weights)))
    axes[1].axhline(1 / 13, color="grey", ls=":", lw=1)
    axes[1].set_xlabel("encoder layer (0 = input to the transformer)")
    axes[1].set_ylabel("learned weight")
    axes[1].set_title("Learned layer weights (dotted = uniform)", fontsize=10)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(plots, "fusion_gate_and_layer_weights.png"), dpi=130)
    plt.close(fig)


def fusion_vs_speech(runs, out):
    def preds(run):
        return {r["id"]: r for r in csv.DictReader(open(os.path.join(run["dir"], "predictions.csv")))}

    rows = []
    for fusion_variant, speech_variant in (("fusion_gated", "speech_wavlm"), ("fusion_concat", "speech_wavlm"),
                                           ("fusion_gated_spknorm", "speech_wavlm_spknorm")):
        speech_runs = {(m["split"], m["seed"]): m for m in select(runs, speech_variant)}
        for m in select(runs, fusion_variant):
            sm = speech_runs.get((m["split"], m["seed"]))
            if not sm:
                continue
            fu, sp = preds(m), preds(sm)
            rows.append({"fusion": fusion_variant, "speech": speech_variant, "split": m["split"], "seed": m["seed"],
                         "speech_acc": round(sm["accuracy"], 4), "fusion_acc": round(m["accuracy"], 4),
                         "fusion_fixed": sum(1 for i in fu if fu[i]["pred"] == fu[i]["true"] and sp[i]["pred"] != sp[i]["true"]),
                         "fusion_broke": sum(1 for i in fu if fu[i]["pred"] != fu[i]["true"] and sp[i]["pred"] == sp[i]["true"]),
                         "mean_gate_speech": round(m["mean_gate_speech"], 4) if "mean_gate_speech" in m else ""})
    with open(os.path.join(out, "fusion_vs_speech.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


PAIRS = [("speech_wavlm", "speech_logmel", "pretrained WavLM vs hand-crafted log-mel"),
         ("fusion_gated", "speech_wavlm", "gated fusion vs speech only"),
         ("fusion_concat", "speech_wavlm", "concat fusion vs speech only"),
         ("speech_wavlm_spknorm", "speech_wavlm", "speaker normalisation, speech model"),
         ("fusion_gated_spknorm", "fusion_gated", "speaker normalisation, gated fusion")]


def paired_comparison(runs, out):
    """Speaker-independent splits: difference A - B for the same split and seed (same data split, same seed)."""
    rows = []
    for a, b, label in PAIRS:
        ra = {(m["split"], m["seed"]): m for m in select(runs, a, "speaker")}
        rb = {(m["split"], m["seed"]): m for m in select(runs, b, "speaker")}
        keys = sorted(set(ra) & set(rb))
        if len(keys) < 2:
            continue
        for metric in ("accuracy", "macro_f1"):
            d = np.array([ra[k][metric] - rb[k][metric] for k in keys]) * 100
            rows.append({"comparison": label, "A": a, "B": b, "metric": metric, "n_pairs": len(keys),
                         "mean_diff_points": round(float(d.mean()), 1), "std_diff": round(float(d.std(ddof=1)), 1),
                         "A_better": int((d > 0).sum()), "min_diff": round(float(d.min()), 1),
                         "max_diff": round(float(d.max()), 1)})
    if not rows:
        return
    with open(os.path.join(out, "paired_seed_comparison.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(out, "paired_seed_comparison.md"), "w", encoding="utf-8") as f:
        f.write("Speaker-independent splits, A minus B on the same split and seed (percentage points).\n\n")
        f.write("| Comparison | Metric | Mean diff ± std | A better in | Range |\n|---|---|---|---|---|\n")
        for r in rows:
            f.write("| {comparison} | {metric} | {mean_diff_points:+.1f} ± {std_diff:.1f} | {A_better}/{n_pairs} | "
                    "{min_diff:+.1f} to {max_diff:+.1f} |\n".format(**r))


def seed_plot(runs, plots):
    variants = [v for v in VARIANT_NAMES if any(m["plan"] == "seeds" for m in select(runs, v, "speaker"))]
    if not variants:
        return
    fig, ax = plt.subplots(figsize=(11, 4))
    for k, split in enumerate(SPEAKER_SPLITS):
        for j, variant in enumerate(variants):
            vals = [100 * m["accuracy"] for m in select(runs, variant, split=split)]
            x = j + (k - 0.5) * 0.3
            ax.scatter([x] * len(vals), vals, color="#1f77b4" if k == 0 else "#d62728", s=18,
                       label=("train OAF, test YAF" if k == 0 else "train YAF, test OAF") if j == 0 else None)
            ax.plot([x - 0.1, x + 0.1], [np.mean(vals)] * 2, color="k", lw=1.5)
    ax.set_xticks(range(len(variants)), [VARIANT_NAMES[v] for v in variants], rotation=15, fontsize=8, ha="right")
    ax.set_ylabel("test accuracy (%)")
    ax.axhline(100 / 7, color="grey", ls=":", lw=1)
    ax.set_title("Speaker-independent accuracy for each seed (dots) and mean (bar); dotted line = chance", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(plots, "speaker_seed_spread.png"), dpi=130)
    plt.close(fig)


def failure_cases(runs, data_dir, out, variant):
    """Misclassified test clips (from the first seed of each split), with how many seeds also got them wrong."""
    manifest = {r["id"]: r for r in csv.DictReader(open(os.path.join(data_dir, "manifest.csv")))}
    chosen = select(runs, variant)
    wrong = defaultdict(int)
    seeds = defaultdict(int)
    for m in chosen:
        seeds[m["split"]] += 1
        for r in csv.DictReader(open(os.path.join(m["dir"], "predictions.csv"))):
            if r["pred"] != r["true"]:
                wrong[(m["split"], r["id"])] += 1
    rows, done = [], set()
    for m in chosen:
        if m["split"] in done:
            continue
        done.add(m["split"])
        for r in csv.DictReader(open(os.path.join(m["dir"], "predictions.csv"))):
            if r["pred"] != r["true"]:
                rows.append({"split": m["split"], "seeds_wrong": "{}/{}".format(wrong[(m["split"], r["id"])], seeds[m["split"]]),
                             "id": r["id"], "speaker": manifest[r["id"]]["speaker"], "word": manifest[r["id"]]["word"],
                             "true": r["true"], "pred": r["pred"], "confidence": float(r["confidence"]),
                             "p_true": float(r["p_" + r["true"]])})
    rows.sort(key=lambda r: -r["confidence"])
    with open(os.path.join(out, "failure_cases_{}.csv".format(variant)), "w", newline="") as f:
        writer = csv.DictWriter(f, ["split", "seeds_wrong", "id", "speaker", "word", "true", "pred", "confidence", "p_true"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def curves(runs, plots):
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
    for ax, variant in zip(axes, ("speech_wavlm", "text_bert", "fusion_gated")):
        for split, colour in (("random_s0", "#1f77b4"), ("word_s0", "#2ca02c"), ("speaker_OAF2YAF", "#d62728")):
            picked = select(runs, variant, split=split)
            if not picked:
                continue
            hist = list(csv.DictReader(open(os.path.join(picked[0]["dir"], "history.csv"))))
            ep = [int(h["epoch"]) for h in hist]
            ax.plot(ep, [float(h["train_loss"]) for h in hist], "-", color=colour, lw=1, label=split + " train")
            ax.plot(ep, [float(h["val_loss"]) for h in hist], "--", color=colour, lw=1, label=split + " val")
        ax.set_title(VARIANT_NAMES[variant], fontsize=10)
        ax.set_xlabel("epoch")
        ax.set_ylabel("cross-entropy")
        ax.set_ylim(0, 2.5)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(plots, "training_curves.png"), dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", nargs="+", default=["runs"])
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--out", default="Results")
    args = parser.parse_args()
    plots = os.path.join(args.out, "plots")
    os.makedirs(plots, exist_ok=True)
    runs = load_runs(args.runs_dir)
    print(len(runs), "runs")
    accuracy_tables(runs, args.out)
    per_class(runs, args.out)
    confusion_plots(runs, plots)
    # The gate histogram and t-SNE need each run's embeddings.npz (written by test.py). Those files are
    # about 190 MB in total, so the repository does not include them; without them these plots are skipped.
    have_embeddings = all(os.path.exists(os.path.join(m["dir"], "embeddings.npz")) for m in runs.values())
    if have_embeddings:
        gate_and_layer_plot(runs, plots)
    else:
        print("embeddings.npz missing: skipping the gate histogram, t-SNE and cluster_separability.csv")
    fusion_vs_speech(runs, args.out)
    paired_comparison(runs, args.out)
    seed_plot(runs, plots)
    curves(runs, plots)
    if have_embeddings:
        tsne_plots(runs, args.data_dir, plots, args.out)
    for variant in ("fusion_gated", "speech_wavlm_spknorm"):
        for r in failure_cases(runs, args.data_dir, args.out, variant)[:5]:
            print(variant, r)
    for name in ("accuracy_table_accuracy.md", "accuracy_speaker_directions.md"):
        print(open(os.path.join(args.out, name), encoding="utf-8").read())


if __name__ == "__main__":
    main()
