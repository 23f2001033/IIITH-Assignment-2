"""Write notebooks/emotion_kaggle.ipynb: prepare data, extract features on the GPU, run every model.

On Kaggle, attach the TESS dataset (ejlok1/toronto-emotional-speech-set-tess) and a dataset holding this
folder as 2b_emotion_recognition.tar.gz; turn on GPU and internet (for WavLM and BERT weights).
"""
import json
import os

CELLS = [
    """# Which run plans to execute: "main" = 5 variants x 8 splits; "speaker_seeds" = speaker-independent
# splits with 5 seeds, with and without per-speaker normalisation
PLANS = ["main", "speaker_seeds"]
import glob, os, shutil, subprocess, sys, tarfile
WORK = "/kaggle/working/2b"
tar = glob.glob("/kaggle/input/**/2b_emotion_recognition.tar.gz", recursive=True)
shutil.rmtree(WORK, ignore_errors=True)
if tar:
    with tarfile.open(tar[0]) as t:
        t.extractall("/kaggle/working/_src")
    shutil.copytree("/kaggle/working/_src/2b_emotion_recognition", WORK)
    shutil.rmtree("/kaggle/working/_src")
else:  # repository cloned instead
    shutil.copytree(glob.glob("/kaggle/input/**/2b_emotion_recognition", recursive=True)[0], WORK)
TESS = [p for p in glob.glob("/kaggle/input/**/*", recursive=True)
        if os.path.isdir(p) and p.rstrip("/").endswith("toronto-emotional-speech-set-tess")][0]
os.chdir(WORK)
subprocess.run("nvidia-smi -L", shell=True)
print(TESS, sorted(os.listdir(WORK)))""",
    """# 1. De-duplicated manifest and splits
subprocess.run([sys.executable, "-m", "common.prepare", "--tess_root", TESS, "--data_dir", "data"], check=True)""",
    """# 2. Preprocessing + frozen feature extraction (WavLM Base+, log-mel, BERT), cached in /tmp
subprocess.run([sys.executable, "-m", "common.features", "--data_dir", "data", "--cache_dir", "/tmp/cache"], check=True)
shutil.copy("/tmp/cache/durations.csv", "data/durations.csv")""",
    """# 3. Training and testing (main: about 40 runs, 40 min on a T4; speaker_seeds: 60 runs, about 60 min)
for plan in PLANS:
    r = subprocess.run([sys.executable, "scripts/run_all.py", "--data_dir", "data", "--cache_dir", "/tmp/cache",
                        "--runs_dir", "runs", "--plan", plan], capture_output=True, text=True)
    print(plan, r.stdout[-20000:]); print(r.stderr[-5000:])""",
    """# 4. Keep only results in the output
out = "/kaggle/working/2b_outputs"
shutil.rmtree(out, ignore_errors=True)
shutil.copytree("data", os.path.join(out, "data"))
shutil.copytree("runs", os.path.join(out, "runs"))
os.chdir("/kaggle/working")
shutil.rmtree(WORK)
print(subprocess.run("du -sh /kaggle/working/2b_outputs", shell=True, capture_output=True, text=True).stdout)""",
]

if __name__ == "__main__":
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                     "source": c.splitlines(keepends=True)} for c in CELLS],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
          "nbformat": 4, "nbformat_minor": 5}
    path = os.path.join(here, "notebooks", "emotion_kaggle.ipynb")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, indent=1)
    print("wrote", path)
