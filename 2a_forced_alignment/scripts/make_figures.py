"""Draw alignment figures with Praat: spectrogram + waveform + tiers from several runs, one PNG each.

For each figure the tiers of the chosen runs are merged into one TextGrid (outputs/comparison/), so
Praat shows the runs one above the other on the same time axis.

usage: python scripts/make_figures.py [--praat D:/Praat/Praat.exe]
"""
import argparse
import glob
import os
import subprocess

from praatio import textgrid

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHORT = {"A_arpa": "A", "B_mfa": "B", "C_adapted": "C", "D_no_g2p": "D"}

# (figure name, file, first word, last word, runs, padding seconds); None words = whole file
FIGURES = [
    ("isle_01_white_bait", "ISLE_SESS0131_BLOCKD02_01_sprt1", None, None, ["A_arpa", "B_mfa", "C_adapted"], 0),
    ("isle_02_new_no", "ISLE_SESS0131_BLOCKD02_02_sprt1", None, None, ["A_arpa", "B_mfa", "C_adapted"], 0),
    ("isle_03_bad_bed", "ISLE_SESS0131_BLOCKD02_03_sprt1", None, None, ["A_arpa", "B_mfa", "C_adapted"], 0),
    ("news1_sjc_hennessy", "F2BJRLP1", "the", "hennessy", ["A_arpa", "B_mfa"], 0.15),
    ("news1_wbur_melnicove", "F2BJRLP1", "as", "reports", ["A_arpa", "B_mfa", "D_no_g2p"], 0.15),
    ("news2_dukakis_depoliticize", "F2BJRLP2", "governor", "judicial", ["A_arpa", "D_no_g2p"], 0.15),
]


def tg_path(run, name):
    found = glob.glob(os.path.join(HERE, "outputs", "textgrids", run, "*", name + ".TextGrid"))
    return found[0] if found else None


def window(name, first, last, pad):
    """Time span from the first occurrence of `first` to the next `last` after it, in run A."""
    tg = textgrid.openTextgrid(tg_path("A_arpa", name), includeEmptyIntervals=False)
    words = tg.getTier("words").entries
    if first is None:
        return 0.0, 0.0
    if name == "F2BJRLP1" and first == "the":  # "the S J C's current leader Edward Hennessy"
        start_index = next(i for i, w in enumerate(words) if w[2] == "the" and words[i + 1][2] == "s")
    else:
        start_index = next(i for i, w in enumerate(words) if w[2] == first)
    end_index = next(i for i, w in enumerate(words) if i >= start_index and w[2] == last)
    return max(0.0, words[start_index][0] - pad), words[end_index][1] + pad


def merged_textgrid(name, runs, out_path):
    merged = textgrid.Textgrid()
    manual = os.path.join(HERE, "manual_labels", name + ".TextGrid")
    sources = ([("manual", manual)] if os.path.exists(manual) else []) + [(r, tg_path(r, name)) for r in runs]
    for run, path in sources:
        if not path:
            continue
        tg = textgrid.openTextgrid(path, includeEmptyIntervals=False)
        tiers = [tg.tierNames[0]] if run == "manual" else ["words", "phones"]
        for tier_name in tiers:
            tier = tg.getTier(tier_name)
            label = "manual words" if run == "manual" else "{} {}".format(SHORT[run], tier_name)
            merged.addTier(textgrid.IntervalTier(label, tier.entries, tg.minTimestamp, tg.maxTimestamp))
    merged.save(out_path, format="long_textgrid", includeBlankSpaces=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--praat", default=r"D:\Praat\Praat.exe")
    args = parser.parse_args()
    comp_dir = os.path.join(HERE, "outputs", "comparison")
    fig_dir = os.path.join(HERE, "figures")
    os.makedirs(comp_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    for fig_name, name, first, last, runs, pad in FIGURES:
        runs = [r for r in runs if tg_path(r, name)]
        merged = os.path.join(comp_dir, fig_name + ".TextGrid")
        merged_textgrid(name, runs, merged)
        start, end = window(name, first, last, pad)
        wav = os.path.join(HERE, "data", "raw", "wav", name + ".wav")
        rows = (["manual"] if os.path.exists(os.path.join(HERE, "manual_labels", name + ".TextGrid")) else []) \
            + ["{0} words, {0} phones".format(SHORT[r]) for r in runs]
        # Praat reads "_" as a subscript marker, so the title avoids underscores
        title = "{}   rows: {}".format(name.replace("_", "-"), " / ".join(rows))
        png = os.path.join(fig_dir, fig_name + ".png")
        subprocess.run([args.praat, "--run", os.path.join(HERE, "scripts", "praat", "draw_alignment.praat"),
                        os.path.abspath(wav), merged, str(round(start, 3)), str(round(end, 3)), png, title],
                       check=True, capture_output=True)
        print("wrote", os.path.relpath(png, HERE), "{:.2f}-{:.2f} s".format(start, end))


if __name__ == "__main__":
    main()
