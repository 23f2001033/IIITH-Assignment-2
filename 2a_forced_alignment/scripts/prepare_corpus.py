"""Turn the provided wav + transcript pairs into an MFA corpus.

MFA expects one folder per speaker, each holding <name>.wav and a transcript <name>.lab with the
same base name. The transcripts are also normalised so that they match what is spoken; every
change is written to normalisation_log.csv so it can be checked.

usage: python scripts/prepare_corpus.py --raw data/raw --out data/corpus
"""
import argparse
import csv
import os
import re
import shutil

import soundfile as sf

# Spoken as letters, so each letter needs to be its own token for the dictionary.
ACRONYMS = {"WBUR"}
# Hyphenated prefixes that are spoken as one word.
JOIN = {"de-politicize": "depoliticize"}


def speaker_of(name):
    """The radio passages share one speaker (f2b); the ISLE files share one learner (session 131)."""
    if name.startswith("F2B"):
        return "f2b"
    if name.startswith("ISLE_SESS"):
        return "isle_" + name.split("_")[1].lower()
    raise ValueError("Unknown speaker for " + name)


def normalise(text):
    """Return the normalised transcript and a list of (rule, before, after) changes."""
    changes = []

    def apply(rule, pattern, repl, flags=0):
        nonlocal text
        for m in re.finditer(pattern, text, flags):
            after = m.expand(repl) if isinstance(repl, str) else repl(m)
            if m.group(0) != after:
                changes.append((rule, m.group(0), after))
        text = re.sub(pattern, repl, text, flags=flags)

    text = text.replace("\r", " ").replace("\n", " ")
    # "seventy- six": a line-wrap hyphen followed by a space; the words are spoken separately.
    apply("split number hyphen", r"(\w+)-\s+(\w+)", r"\1 \2")
    for before, after in JOIN.items():
        apply("join prefix", re.escape(before), after)
    # S.J.C.'s -> S J C's : dotted initialisms are read letter by letter.
    apply("dotted initialism", r"\b((?:[A-Z]\.){2,})('s)?",
          lambda m: " ".join(m.group(1).replace(".", " ").split()) + (m.group(2) or ""))
    for word in ACRONYMS:
        apply("spoken acronym", r"\b%s('s)?\b" % word, lambda m, w=word: " ".join(w) + (m.group(1) or ""))
    # Punctuation is not spoken; apostrophes inside words are kept (Hennessy's, state's).
    apply("drop punctuation", r"[^\w\s']|(?<!\w)'|'(?!\w)", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text, changes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if os.path.isdir(args.out):
        shutil.rmtree(args.out)
    rows = []
    log = []
    for wav_name in sorted(os.listdir(os.path.join(args.raw, "wav"))):
        name = os.path.splitext(wav_name)[0]
        transcript = [f for f in os.listdir(os.path.join(args.raw, "transcripts"))
                      if os.path.splitext(f)[0] == name]
        assert len(transcript) == 1, (name, transcript)
        with open(os.path.join(args.raw, "transcripts", transcript[0]), encoding="utf-8") as f:
            raw_text = f.read()
        text, changes = normalise(raw_text)
        speaker = speaker_of(name)
        folder = os.path.join(args.out, speaker)
        os.makedirs(folder, exist_ok=True)
        info = sf.info(os.path.join(args.raw, "wav", wav_name))
        assert info.samplerate == 16000 and info.channels == 1, (wav_name, info)
        shutil.copy(os.path.join(args.raw, "wav", wav_name), os.path.join(folder, wav_name))
        with open(os.path.join(folder, name + ".lab"), "w", encoding="utf-8", newline="\n") as f:
            f.write(text + "\n")
        rows.append([name, speaker, round(info.duration, 2), len(text.split()), text])
        log += [[name, rule, before, after] for rule, before, after in changes if rule != "drop punctuation"]

    with open(os.path.join(args.out, "corpus_summary.csv"), "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows([["file", "speaker", "seconds", "words", "normalised_text"]] + rows)
    with open(os.path.join(args.out, "normalisation_log.csv"), "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows([["file", "rule", "before", "after"]] + log)
    for row in rows:
        print(row[0], row[1], row[2], "s,", row[3], "words")
    for row in log:
        print("  ", row)


if __name__ == "__main__":
    main()
