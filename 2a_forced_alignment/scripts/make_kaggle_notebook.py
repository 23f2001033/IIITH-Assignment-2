"""Write notebooks/mfa_alignment_kaggle.ipynb, which runs scripts/run_pipeline.sh on Kaggle (CPU).

The notebook expects this folder (2a_forced_alignment) attached as a Kaggle dataset, either as a
folder or as 2a_forced_alignment.tar.gz, and writes everything to /kaggle/working/2a_outputs.
"""
import json
import os

CELLS = [
    """# Forced alignment with Montreal Forced Aligner, run on Kaggle (CPU, internet on).
# Attach the dataset that holds 2a_forced_alignment, then Run All.
import glob, os, shutil, subprocess, tarfile
WORK_DIR = "/kaggle/working/2a"
found = glob.glob("/kaggle/input/**/2a_forced_alignment.tar.gz", recursive=True)
if found:
    os.makedirs(WORK_DIR, exist_ok=True)
    with tarfile.open(found[0]) as t:
        t.extractall("/kaggle/working/_src")
    src = glob.glob("/kaggle/working/_src/**/scripts/run_pipeline.sh", recursive=True)[0]
else:
    src = glob.glob("/kaggle/input/**/scripts/run_pipeline.sh", recursive=True)[0]
src = os.path.dirname(os.path.dirname(src))
shutil.rmtree(WORK_DIR, ignore_errors=True)
shutil.copytree(src, WORK_DIR, ignore=shutil.ignore_patterns("outputs", "corpus"))
print(sorted(os.listdir(WORK_DIR)))""",
    """# Install MFA from conda-forge into a micromamba environment (about 3 minutes)
MAMBA = "/tmp/bin/micromamba"
if not os.path.exists(MAMBA):
    subprocess.run("curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C /tmp bin/micromamba",
                   shell=True, check=True)
    subprocess.run(MAMBA + " create -y -q -r /tmp/mamba -n mfa -c conda-forge montreal-forced-aligner",
                   shell=True, check=True)
MFA = MAMBA + " run -r /tmp/mamba -n mfa mfa"
print(subprocess.run(MFA + " version", shell=True, capture_output=True, text=True).stdout)""",
    """# Run the whole pipeline: corpus, dictionaries + G2P, four alignment runs, adaptation
env = dict(os.environ, MFA=MFA, PY="python", MFA_ROOT_DIR="/kaggle/working/mfa_root", WORK="/tmp/mfa_work")
result = subprocess.run("bash scripts/run_pipeline.sh", shell=True, cwd=WORK_DIR, env=env,
                        capture_output=True, text=True)
print(result.stdout[-15000:])
print("STDERR:", result.stderr[-5000:])""",
    """# Collect outputs (TextGrids, logs, dictionaries, MFA's own alignment analysis) in one folder
out = "/kaggle/working/2a_outputs"
shutil.rmtree(out, ignore_errors=True)
shutil.copytree(os.path.join(WORK_DIR, "outputs"), os.path.join(out, "outputs"))
shutil.copytree(os.path.join(WORK_DIR, "dictionaries"), os.path.join(out, "dictionaries"))
shutil.copytree(os.path.join(WORK_DIR, "data", "corpus"), os.path.join(out, "corpus"),
                ignore=shutil.ignore_patterns("*.wav"))
shutil.rmtree("/kaggle/working/mfa_root", ignore_errors=True)
shutil.rmtree("/kaggle/working/_src", ignore_errors=True)
shutil.rmtree(WORK_DIR, ignore_errors=True)
for root, _, files in os.walk(out):
    print(root, len(files))""",
]


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    nb = {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                   "source": cell.splitlines(keepends=True)} for cell in CELLS],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    path = os.path.join(here, "notebooks", "mfa_alignment_kaggle.ipynb")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
