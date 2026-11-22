"""Audit the Kaggle copy of TESS before any modelling: files, duplicates, labels, words, speakers, audio.

usage: python scripts/audit_tess.py --root /kaggle/input/toronto-emotional-speech-set-tess --out audit
"""
import argparse
import collections
import hashlib
import json
import os

import soundfile as sf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for dirpath, _, files in os.walk(args.root):
        for name in files:
            if name.lower().endswith(".wav"):
                path = os.path.join(dirpath, name)
                with open(path, "rb") as f:
                    digest = hashlib.md5(f.read()).hexdigest()
                info = sf.info(path)
                stem = os.path.splitext(name)[0]
                parts = stem.split("_")
                rows.append({"path": os.path.relpath(path, args.root), "folder": os.path.basename(dirpath),
                             "name": name, "md5": digest, "speaker": parts[0].upper(),
                             "word": "_".join(parts[1:-1]).lower(), "emotion_from_name": parts[-1].lower(),
                             "rate": info.samplerate, "channels": info.channels,
                             "seconds": round(info.duration, 3)})

    by_hash = collections.defaultdict(list)
    for r in rows:
        by_hash[r["md5"]].append(r["path"])
    folders = collections.Counter(os.path.dirname(r["path"]).split(os.sep)[0] for r in rows)
    report = {
        "wav_files": len(rows),
        "unique_contents": len(by_hash),
        "duplicate_groups": sum(1 for v in by_hash.values() if len(v) > 1),
        "duplicate_examples": [v for v in by_hash.values() if len(v) > 1][:5],
        "top_level_folders": folders,
        "leaf_folders": collections.Counter(r["folder"] for r in rows),
        "speakers": collections.Counter(r["speaker"] for r in rows),
        "emotions_from_name": collections.Counter(r["emotion_from_name"] for r in rows),
        "words": len({r["word"] for r in rows}),
        "folder_vs_name_mismatch": [r["path"] for r in rows
                                    if r["emotion_from_name"][:3] not in r["folder"].lower().replace("pleasant_surprise", "ps")][:20],
        "sample_rates": collections.Counter(r["rate"] for r in rows),
        "channels": collections.Counter(r["channels"] for r in rows),
        "seconds_min_median_max": sorted(r["seconds"] for r in rows)[::max(1, len(rows) // 2)][:3]
        if rows else [],
    }
    secs = sorted(r["seconds"] for r in rows)
    report["seconds_min_median_max"] = [secs[0], secs[len(secs) // 2], secs[-1]]
    unique = {}
    for r in rows:
        unique.setdefault(r["md5"], r)
    combos = collections.Counter((r["speaker"], r["emotion_from_name"]) for r in unique.values())
    report["unique_per_speaker_emotion"] = {"{}_{}".format(*k): v for k, v in sorted(combos.items())}
    words_per_emotion = collections.defaultdict(set)
    for r in unique.values():
        words_per_emotion[r["emotion_from_name"]].add(r["word"])
    report["words_per_emotion"] = {k: len(v) for k, v in words_per_emotion.items()}
    with open(os.path.join(args.out, "audit.json"), "w") as f:
        json.dump(report, f, indent=1, default=dict)
    with open(os.path.join(args.out, "files.tsv"), "w") as f:
        keys = list(rows[0])
        f.write("\t".join(keys) + "\n" + "".join("\t".join(str(r[k]) for k in keys) + "\n" for r in rows))
    print(json.dumps(report, indent=1, default=dict))


if __name__ == "__main__":
    main()
