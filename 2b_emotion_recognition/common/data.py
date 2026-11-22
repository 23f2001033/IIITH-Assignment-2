"""TESS manifest and data splits.

The Kaggle copy of TESS stores every file twice (a nested copy of the whole folder), so files are
de-duplicated by content before anything else. Speaker and emotion come from the folder name, because
one file name has a typo (OA_bite_neutral.wav). The text of every clip is the carrier sentence
"Say the word <word>", which TESS uses for all 2800 recordings.

Three kinds of split are produced:
  random_s<k>     stratified by speaker x emotion, 70/15/15. The usual protocol; the same word, said
                  by the same speaker in other emotions, appears in train and test.
  word_s<k>       word-disjoint: all 14 recordings of a word (2 speakers x 7 emotions) land in one split,
                  140/30/30 words. Tests generalisation to unseen words.
  speaker_<A>2<B> train and validate on speaker A (validation words held out), test on speaker B.
                  Tests generalisation to an unseen voice, the hardest condition with two speakers.
"""
import csv
import hashlib
import os
import random
from collections import defaultdict

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "pleasant_surprise", "sad"]
_ALIASES = {"pleasant_surprise": "pleasant_surprise", "pleasant_surprised": "pleasant_surprise", "ps": "pleasant_surprise"}


def _emotion_from_folder(folder):
    raw = folder.split("_", 1)[1].lower()
    return _ALIASES.get(raw, raw)


def build_manifest(root):
    """One row per unique recording: id, path, speaker, word, emotion, text."""
    seen, rows = {}, []
    for dirpath, _, files in sorted(os.walk(root)):
        for name in sorted(files):
            if not name.lower().endswith(".wav"):
                continue
            path = os.path.join(dirpath, name)
            with open(path, "rb") as f:
                digest = hashlib.md5(f.read()).hexdigest()
            if digest in seen:
                continue
            folder = os.path.basename(dirpath)
            speaker = folder.split("_")[0].upper()
            emotion = _emotion_from_folder(folder)
            word = os.path.splitext(name)[0].split("_")[1].lower()
            seen[digest] = path
            rows.append({"id": "{}_{}_{}".format(speaker, word, emotion), "path": path, "speaker": speaker,
                         "word": word, "emotion": emotion, "text": "Say the word {}".format(word)})
    assert len({r["id"] for r in rows}) == len(rows), "ids are not unique"
    return rows


def make_splits(rows, seeds=(0, 1, 2)):
    """{split_name: {id: 'train' | 'val' | 'test'}}"""
    splits = {}
    for seed in seeds:
        rng = random.Random(seed)
        assign = {}
        groups = defaultdict(list)
        for r in rows:
            groups[(r["speaker"], r["emotion"])].append(r["id"])
        for ids in groups.values():
            ids = sorted(ids)
            rng.shuffle(ids)
            n_test = n_val = round(0.15 * len(ids))
            for i, rid in enumerate(ids):
                assign[rid] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
        splits["random_s{}".format(seed)] = assign

        words = sorted({r["word"] for r in rows})
        rng = random.Random(seed)
        rng.shuffle(words)
        n = len(words)
        part = {w: ("test" if i < round(0.15 * n) else "val" if i < round(0.30 * n) else "train")
                for i, w in enumerate(words)}
        splits["word_s{}".format(seed)] = {r["id"]: part[r["word"]] for r in rows}

    for train_spk, test_spk in (("OAF", "YAF"), ("YAF", "OAF")):
        words = sorted({r["word"] for r in rows})
        rng = random.Random(0)
        rng.shuffle(words)
        val_words = set(words[:round(0.15 * len(words))])
        assign = {}
        for r in rows:
            if r["speaker"] == test_spk:
                assign[r["id"]] = "test"
            else:
                assign[r["id"]] = "val" if r["word"] in val_words else "train"
        splits["speaker_{}2{}".format(train_spk, test_spk)] = assign
    return splits


def write_csv(path, rows, fields):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_dataset(root, out_dir):
    """Write manifest.csv and splits.csv (one column per split) to out_dir."""
    rows = build_manifest(root)
    splits = make_splits(rows)
    write_csv(os.path.join(out_dir, "manifest.csv"), rows, ["id", "path", "speaker", "word", "emotion", "text"])
    table = [dict({"id": r["id"]}, **{name: s[r["id"]] for name, s in splits.items()}) for r in rows]
    write_csv(os.path.join(out_dir, "splits.csv"), table, ["id"] + list(splits))
    return rows, splits


def load_split(data_dir, split):
    """{'train': [ids], 'val': [...], 'test': [...]} for one split name."""
    out = {"train": [], "val": [], "test": []}
    for row in read_csv(os.path.join(data_dir, "splits.csv")):
        out[row[split]].append(row["id"])
    return out


def labels(data_dir):
    return {r["id"]: EMOTIONS.index(r["emotion"]) for r in read_csv(os.path.join(data_dir, "manifest.csv"))}
