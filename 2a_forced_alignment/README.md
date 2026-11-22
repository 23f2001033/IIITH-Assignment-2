# Assignment 2a: Forced alignment with the Montreal Forced Aligner

Six provided recordings are aligned at word and phone level with MFA 3.4.4, using two pretrained
acoustic models, a speaker-adapted model, and a custom dictionary built with G2P plus hand checks.
The resulting TextGrids are inspected in Praat and compared across models, and the learner's vowels are measured
inside the aligned intervals.

| Data | Files | Speaker | Duration |
|---|---|---|---|
| Radio news passages (`F2BJRLP1-3`) | 3 | one professional female newsreader | 25-31 s each |
| ISLE learner prompts ("I SAID WHITE NOT BAIT", "NEW NOT NO", "BAD NOT BED") | 3 | one non-native adult male learner | about 4 s each |

All files are 16 kHz mono and are used as provided.

## Layout

```
data/raw/            the provided wav files and transcripts, unchanged
data/corpus/         MFA corpus: one folder per speaker, <file>.wav + normalised <file>.lab
dictionaries/        manual_*.txt (hand corrections to G2P), oov_final_*.dict (pronunciations used for OOV words)
scripts/
  prepare_corpus.py      transcript normalisation and MFA folder layout (logs every change)
  find_oovs.py           words missing from a dictionary
  apply_overrides.py     merges base dictionary + G2P output + hand corrections, checks phone inventory
  run_pipeline.sh        the whole MFA pipeline in one command
  analyse_alignments.py  tables, model agreement, error flags, vowel formants, and accuracy against hand labels if `manual_labels/` holds any
  make_figures.py        Praat figures (spectrogram + waveform + tiers of several runs)
  praat/                 Praat scripts used headless by the two scripts above
notebooks/           Kaggle notebook that runs run_pipeline.sh (CPU only)
manual_labels/       (optional, create it) hand-labelled Praat TextGrids with the same file names; every run is scored against them
outputs/
  textgrids/<run>/   MFA TextGrids for each run, plus MFA's alignment_analysis.csv
  analysis/          CSV results of analyse_alignments.py
  comparison/        merged TextGrids used for the figures (open them in Praat)
  logs/              MFA logs
figures/             PNG figures used in the report
```

## Alignment runs

| Run | Acoustic model | Dictionary | Purpose |
|---|---|---|---|
| A_arpa | `english_us_arpa` | `english_us_arpa` + checked G2P entries | main alignment |
| B_mfa | `english_mfa` | `english_us_mfa` + checked G2P entries | second acoustic model and phone set |
| C_adapted | `english_us_arpa` adapted to the two speakers (`mfa adapt`) | as A | does speaker adaptation move boundaries? |
| D_no_g2p | `english_us_arpa` | base dictionary only | effect of leaving OOV words unknown (`spn`) |
| A_default_beam | as A, MFA's default beam | as A | are the wide beam settings needed for 30 s files? |

## How to run

### 1. Install MFA

MFA is distributed through conda-forge (it needs Kaldi and OpenFST binaries, so `pip` alone is not enough):

```bash
conda create -n mfa -c conda-forge montreal-forced-aligner
conda activate mfa
mfa version            # 3.4.4 was used
pip install praatio matplotlib soundfile numpy   # for the analysis scripts
```

Praat (https://www.fon.hum.uva.nl/praat/) is needed only for the figures and formant measurements.

### 2. Run the whole pipeline

From this folder (`2a_forced_alignment/`):

```bash
bash scripts/run_pipeline.sh
```

This prepares the corpus, downloads the pretrained models, validates the corpus, builds the custom
dictionaries, and runs the five alignments into `outputs/textgrids/`. The individual steps are:

```bash
python scripts/prepare_corpus.py --raw data/raw --out data/corpus
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa
mfa model download g2p english_us_arpa
mfa validate data/corpus english_us_arpa --acoustic_model_path english_us_arpa
python scripts/find_oovs.py --corpus data/corpus --dictionary <MFA_ROOT>/pretrained_models/dictionary/english_us_arpa.dict --out oovs.txt
mfa g2p oovs.txt english_us_arpa g2p_arpa.dict --num_pronunciations 1
python scripts/apply_overrides.py --base <english_us_arpa.dict> --g2p g2p_arpa.dict \
    --overrides dictionaries/manual_arpa.txt --out custom_arpa.dict --report dictionaries/oov_final_arpa.dict
mfa align data/corpus custom_arpa.dict english_us_arpa outputs/textgrids/A_arpa --beam 100 --retry_beam 400 --clean
mfa adapt data/corpus custom_arpa.dict english_us_arpa adapted_arpa.zip --beam 100 --retry_beam 400 --clean
mfa align data/corpus custom_arpa.dict adapted_arpa.zip outputs/textgrids/C_adapted --beam 100 --retry_beam 400 --clean
```

On Kaggle, `notebooks/mfa_alignment_kaggle.ipynb` installs MFA with micromamba and runs the same script.

### 3. Analyse and draw

```bash
python scripts/analyse_alignments.py --praat "D:/Praat/Praat.exe"
python scripts/make_figures.py --praat "D:/Praat/Praat.exe"
```

To inspect an alignment interactively, open the wav and its TextGrid in Praat, select both and click
**View & Edit**. The merged TextGrids in `outputs/comparison/` show several runs at once.
