"""List corpus words that a pronunciation dictionary does not contain.

MFA lower-cases words and looks them up whole, so the same is done here. The list feeds
`mfa g2p`, which generates pronunciations for exactly these words.

usage: python scripts/find_oovs.py --corpus data/corpus --dictionary english_us_arpa.dict --out oovs.txt
"""
import argparse
import glob
import os
from collections import Counter


def dictionary_words(path):
    words = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                words.add(line.split()[0].lower())
    return words


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--dictionary", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    known = dictionary_words(args.dictionary)
    counts = Counter()
    for lab in glob.glob(os.path.join(args.corpus, "*", "*.lab")):
        with open(lab, encoding="utf-8") as f:
            counts.update(w.lower() for w in f.read().split())
    oovs = sorted(w for w in counts if w not in known)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write("".join(w + "\n" for w in oovs))
    print("{} word types, {} not in {}: {}".format(len(counts), len(oovs), os.path.basename(args.dictionary),
                                                  " ".join(oovs)))


if __name__ == "__main__":
    main()
