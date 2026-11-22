"""Analyse the MFA alignments: tables, model agreement, error flags, manual-label accuracy, vowels.

Reads outputs/textgrids/<run>/<speaker>/<file>.TextGrid for every run and writes:
  outputs/analysis/alignments_long.csv   every word and phone interval of every run
  outputs/analysis/agreement.csv         word-boundary differences between runs (ms)
  outputs/analysis/flags.csv             intervals worth checking in Praat
  outputs/analysis/manual_eval.csv       boundary error against hand labels (if manual_labels/ has them)
  outputs/analysis/formants.csv          F1/F2 of the ISLE target vowels (measured by Praat)
  figures/*.png                          plots of the above

usage: python scripts/analyse_alignments.py [--praat D:/Praat/Praat.exe]
"""
import argparse
import csv
import difflib
import glob
import os
import subprocess
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from praatio import textgrid  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = {
    "A_arpa": "A: english_us_arpa + G2P",
    "B_mfa": "B: english_mfa + G2P",
    "C_adapted": "C: english_us_arpa adapted",
    "D_no_g2p": "D: english_us_arpa, no G2P",
    "A_default_beam": "A with default beam",
}
SILENCE = {"", "sil", "sp"}
MIN_PHONE_MS = 30          # MFA uses 10 ms frames and 3-state phone HMMs, so no phone can be shorter
LONG_PAUSE_MS = 300        # pauses longer than this inside a sentence are flagged
# Target vowel of each ISLE minimal-pair word, in each phone set
ISLE_TARGETS = {"white": "AY1", "bait": "EY1", "new": "UW1", "no": "OW1", "bad": "AE1", "bed": "EH1", "not": "AA1"}
# Hillenbrand et al. (1995) adult male averages, used only as a rough reference frame (F1, F2 in Hz)
REFERENCE = {"EY": (476, 2089), "EH": (580, 1799), "AE": (588, 1952), "UW": (378, 997), "OW": (497, 910),
             "AA": (768, 1333)}


def group_of(name):
    return "news" if name.startswith("F2B") else "isle"


def load_run(run):
    """{file: {"words": [(start, end, label)], "phones": [...]}} without silence intervals."""
    out = {}
    for path in sorted(glob.glob(os.path.join(HERE, "outputs", "textgrids", run, "*", "*.TextGrid"))):
        tg = textgrid.openTextgrid(path, includeEmptyIntervals=False)
        name = os.path.splitext(os.path.basename(path))[0]
        out[name] = {tier: [(s, e, t) for s, e, t in tg.getTier(tier).entries if t not in SILENCE]
                     for tier in ("words", "phones")}
    return out


def load_manual():
    out = {}
    for path in sorted(glob.glob(os.path.join(HERE, "manual_labels", "*.TextGrid"))):
        tg = textgrid.openTextgrid(path, includeEmptyIntervals=False)
        tier = tg.getTier(tg.tierNames[0])
        words = [(s, e, t.strip().lower()) for s, e, t in tier.entries if t.strip()]
        if words:  # an unlabelled template contributes nothing
            out[os.path.splitext(os.path.basename(path))[0]] = words
    return out


def matched_boundaries(ref, hyp):
    """Pair up word intervals with the same label in order; return (ref_time, hyp_time, kind, word)."""
    a = [w[2].lower() for w in ref]
    b = [w[2].lower() for w in hyp]
    pairs = []
    for block in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_matching_blocks():
        for k in range(block.size):
            r, h = ref[block.a + k], hyp[block.b + k]
            pairs.append((r[0], h[0], "start", r[2]))
            pairs.append((r[1], h[1], "end", r[2]))
    return pairs


def write_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows([header] + rows)


def summary(diffs_ms):
    d = np.abs(np.array(diffs_ms, dtype=float))
    return [len(d), round(float(np.mean(d)), 1), round(float(np.median(d)), 1), round(float(np.percentile(d, 90)), 1),
            round(float(np.max(d)), 1), round(100 * float(np.mean(d <= 20)), 1), round(100 * float(np.mean(d <= 50)), 1)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--praat", default=r"D:\Praat\Praat.exe")
    args = parser.parse_args()
    out_dir = os.path.join(HERE, "outputs", "analysis")
    fig_dir = os.path.join(HERE, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    runs = {run: load_run(run) for run in RUNS if os.path.isdir(os.path.join(HERE, "outputs", "textgrids", run))}

    # 1. Everything in one long table
    rows = []
    for run, files in runs.items():
        for name, tiers in files.items():
            for tier, items in tiers.items():
                rows += [[run, name, tier, t, round(s, 3), round(e, 3), round(1000 * (e - s))] for s, e, t in items]
    write_csv(os.path.join(out_dir, "alignments_long.csv"),
              ["run", "file", "tier", "label", "start", "end", "duration_ms"], rows)

    # 2. Agreement between runs: how far each word boundary moves when the model or dictionary changes
    agreement, hist = [], {}
    for other in ("B_mfa", "C_adapted", "D_no_g2p", "A_default_beam"):
        if other not in runs:
            continue
        for group in ("news", "isle"):
            diffs = []
            for name in runs["A_arpa"]:
                if group_of(name) == group:
                    diffs += [1000 * (h - r) for r, h, _, _ in
                              matched_boundaries(runs["A_arpa"][name]["words"], runs[other][name]["words"])]
            agreement.append(["A_arpa vs " + other, group] + summary(diffs))
            hist[(other, group)] = diffs
    write_csv(os.path.join(out_dir, "agreement.csv"),
              ["comparison", "files", "boundaries", "mean_abs_ms", "median_abs_ms", "p90_abs_ms", "max_abs_ms",
               "pct_within_20ms", "pct_within_50ms"], agreement)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=False)
    for ax, group in zip(axes, ("news", "isle")):
        for other, colour in (("B_mfa", "#d62728"), ("C_adapted", "#2ca02c"), ("D_no_g2p", "#1f77b4")):
            if (other, group) in hist:
                d = np.abs(hist[(other, group)])
                ax.hist(np.clip(d, 0, 100), bins=np.arange(0, 105, 5), histtype="step", lw=1.8, color=colour,
                        label="A vs " + other.split("_")[0] + " (" + RUNS[other].split(": ")[1] + ")")
        ax.set_title({"news": "Radio news (3 files)", "isle": "ISLE learner (3 files)"}[group])
        ax.set_xlabel("|word boundary difference| (ms, clipped at 100)")
    axes[0].set_ylabel("boundaries")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "model_agreement.png"), dpi=150)
    plt.close(fig)

    # 3. Flags: phones at the 30 ms floor, long pauses inside sentences, unknown-word (spn) segments
    flags = []
    for run in ("A_arpa", "B_mfa", "D_no_g2p"):
        if run not in runs:
            continue
        for name, tiers in runs[run].items():
            for s, e, t in tiers["phones"]:
                if round(1000 * (e - s)) <= MIN_PHONE_MS:
                    word = next((w[2] for w in tiers["words"] if w[0] <= s < w[1]), "")
                    flags.append([run, name, "phone at 30 ms floor", t, word, round(s, 3), round(e, 3)])
                if t == "spn":
                    word = next((w[2] for w in tiers["words"] if w[0] <= s < w[1]), "")
                    flags.append([run, name, "unknown word (spn)", t, word, round(s, 3), round(e, 3)])
            words = tiers["words"]
            for prev, nxt in zip(words, words[1:]):
                gap = 1000 * (nxt[0] - prev[1])
                if gap >= LONG_PAUSE_MS:
                    flags.append([run, name, "pause >= 300 ms", "", prev[2] + " | " + nxt[2],
                                  round(prev[1], 3), round(nxt[0], 3)])
    write_csv(os.path.join(out_dir, "flags.csv"), ["run", "file", "flag", "phone", "word", "start", "end"], flags)

    # Phone duration distribution (run A), with the 30 ms floor marked
    durations = {g: [1000 * (e - s) for n, t in runs["A_arpa"].items() if group_of(n) == g
                     for s, e, _ in t["phones"]] for g in ("news", "isle")}
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for g, colour in (("news", "#1f77b4"), ("isle", "#ff7f0e")):
        ax.hist(durations[g], bins=np.arange(0, 400, 10), alpha=0.6, color=colour, density=True,
                label="{} ({} phones, median {:.0f} ms)".format(g, len(durations[g]), np.median(durations[g])))
    ax.axvline(MIN_PHONE_MS, color="k", ls="--", lw=1)
    ax.text(MIN_PHONE_MS + 3, ax.get_ylim()[1] * 0.9, "30 ms floor", fontsize=8)
    ax.set_xlabel("phone duration (ms), run A")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "phone_durations.png"), dpi=150)
    plt.close(fig)

    # 4. Accuracy against hand labels
    manual = load_manual()
    if manual:
        rows, per_run = [], {}
        for run in runs:
            for name, ref in manual.items():
                if name not in runs[run]:
                    continue
                for r, h, kind, word in matched_boundaries(ref, runs[run][name]["words"]):
                    rows.append([run, name, word, kind, round(r, 3), round(h, 3), round(1000 * (h - r), 1)])
                    per_run.setdefault(run, []).append(1000 * (h - r))
        write_csv(os.path.join(out_dir, "manual_eval.csv"),
                  ["run", "file", "word", "boundary", "manual_s", "mfa_s", "error_ms"], rows)
        write_csv(os.path.join(out_dir, "manual_eval_summary.csv"),
                  ["run", "boundaries", "mean_abs_ms", "median_abs_ms", "p90_abs_ms", "max_abs_ms",
                   "pct_within_20ms", "pct_within_50ms", "mean_signed_ms"],
                  [[run] + summary(d) + [round(float(np.mean(d)), 1)] for run, d in per_run.items()])

    # 5. ISLE target vowels: F1/F2 measured by Praat at 25/50/75% of the aligned vowel
    vowel_rows = []
    for name, tiers in runs["A_arpa"].items():
        if group_of(name) != "isle":
            continue
        intervals = []
        for ws, we, word in tiers["words"]:
            target = ISLE_TARGETS.get(word.lower())
            for s, e, p in tiers["phones"]:
                if target and p == target and ws <= s < we:
                    intervals.append((word.upper() + ":" + p, s, e))
        wav = glob.glob(os.path.join(HERE, "data", "raw", "wav", name + ".wav"))[0]
        with tempfile.TemporaryDirectory() as tmp:
            tsv = os.path.join(tmp, "intervals.tsv")
            with open(tsv, "w", encoding="utf-8") as f:
                f.write("label\tstart\tend\n" + "".join("{}\t{:.4f}\t{:.4f}\n".format(*i) for i in intervals))
            result = subprocess.run([args.praat, "--run", os.path.join(HERE, "scripts", "praat", "measure_formants.praat"),
                                     os.path.abspath(wav), tsv, "5000"], check=True, capture_output=True)
            raw = result.stdout
            text = raw.decode("utf-16-le") if raw[1:2] == b"\x00" else raw.decode("utf-8")
            for row in list(csv.reader(text.strip().splitlines()))[1:]:
                vowel_rows.append([name] + row)
    write_csv(os.path.join(out_dir, "formants.csv"),
              ["file", "label", "start", "end", "duration_ms", "f0", "f1_25", "f2_25", "f1_50", "f2_50", "f1_75", "f2_75"],
              vowel_rows)

    fig, ax = plt.subplots(figsize=(6, 4.6))
    for vowel, (f1, f2) in REFERENCE.items():
        ax.scatter(f2, f1, marker="x", color="grey")
        ax.annotate(vowel + " (ref)", (f2, f1), fontsize=7, color="grey", xytext=(3, 3), textcoords="offset points")
    for row in vowel_rows:
        label, f1s, f2s = row[1], [row[6], row[8], row[10]], [row[7], row[9], row[11]]
        try:
            f1v = [float(v) for v in f1s]
            f2v = [float(v) for v in f2s]
        except ValueError:
            continue
        ax.plot(f2v, f1v, "-", color="#1f77b4", lw=1)
        ax.scatter(f2v[1], f1v[1], color="#1f77b4")
        ax.annotate(label.split(":")[0], (f2v[1], f1v[1]), fontsize=8, xytext=(4, -10), textcoords="offset points")
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel("F2 (Hz)")
    ax.set_ylabel("F1 (Hz)")
    ax.set_title("ISLE target vowels: midpoint (dot), 25-75% track (line)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "isle_vowel_chart.png"), dpi=150)
    plt.close(fig)

    for row in agreement:
        print(row)
    print(len(flags), "flags;", len(vowel_rows), "vowels measured;", "manual labels:", sorted(manual) or "none yet")


if __name__ == "__main__":
    main()
