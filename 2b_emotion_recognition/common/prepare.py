"""Build the de-duplicated manifest and the split table.

usage: python -m common.prepare --tess_root /kaggle/input/toronto-emotional-speech-set-tess --data_dir data
"""
import argparse
from collections import Counter

from common.data import save_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tess_root", required=True)
    parser.add_argument("--data_dir", required=True)
    args = parser.parse_args()
    rows, splits = save_dataset(args.tess_root, args.data_dir)
    print(len(rows), "unique recordings;", Counter((r["speaker"], r["emotion"]) for r in rows).most_common(3), "...")
    for name, assign in splits.items():
        print(name, Counter(assign.values()))


if __name__ == "__main__":
    main()
