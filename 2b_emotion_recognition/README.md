# Assignment 2b: Multimodal emotion recognition on TESS

Speech-only, text-only and speech + text (fusion) emotion classifiers on the Toronto Emotional Speech Set,
built from the five blocks in the brief: preprocessing, feature extraction, temporal / contextual modelling,
fusion, classifier. Every model is trained and tested on three kinds of split, and the hardest one
(an unseen speaker) is repeated with 5 seeds so that differences between models can be told apart from
training noise.

## Data

TESS: two actresses (OAF, aged 64; YAF, aged 26) each say "Say the word _X_" for the same 200 words in
7 emotions (angry, disgust, fear, happy, neutral, pleasant surprise, sad).

An audit before modelling (`scripts/audit_tess.py`, summary in `data/tess_audit.json`) found:

| Finding | Handling |
|---|---|
| The Kaggle copy has 5,600 wav files but only 2,800 distinct recordings: the whole set is duplicated in a nested folder | de-duplicated by content hash (`common/data.py`); duplicates would otherwise leak between train and test |
| Two files are 96 kHz; the rest are 24,414 Hz | everything is resampled to 16 kHz |
| One file is named `OA_bite_neutral.wav` instead of `OAF_...` | speaker and emotion are taken from the folder name, not the file name |
| Clips are 1.25-2.99 s long and start and end with silence | silence is trimmed |

**The text carries no emotion.** Every word appears in all 7 emotions, and the sentence is always
"Say the word _X_". A text-only model therefore cannot beat chance (1/7 = 14.3%). It is built anyway
because the brief asks for it, and because it shows directly what fusion can and cannot gain from text here.

## Splits

`data/splits.csv` assigns every clip to train / val / test for 8 splits:

| Split | What is held out | Train / val / test clips | Seeds |
|---|---|---|---|
| `random_s0-2` | random 15% of clips (stratified by speaker x emotion) | 1960 / 420 / 420 | 3 splits |
| `word_s0-2` | 30 words never seen in training (the same word is in train for no emotion) | 1960 / 420 / 420 | 3 splits |
| `speaker_OAF2YAF`, `speaker_YAF2OAF` | the other speaker entirely (val = 15% of the training speaker's words) | 1190 / 210 / 1400 | 5 training seeds each |

The random split is the setting most often reported for TESS, but it lets the same speaker saying the same
word in another emotion appear in training. The word split rules out memorising words, and the speaker
split asks the real question: does the model recognise emotion in a voice it has never heard?

## Model: the five blocks

| Block | Speech | Text | Why |
|---|---|---|---|
| Preprocessing | mono, resample to 16 kHz, trim silence (`top_db=30`), peak-normalise | lower-case, BERT WordPiece tokens | the encoders expect 16 kHz; silence carries no emotion and differs between clips |
| Feature extraction | frozen **WavLM Base+**, learned softmax-weighted sum of all 13 layers (baseline: 80-band **log-mel**) | frozen **BERT-base-uncased**, weighted sum of 13 layers | 2,800 clips are too few to fine-tune a 95M-parameter encoder; the layer weights let the model choose which depth is useful instead of guessing one layer |
| Temporal / contextual modelling | LayerNorm, linear projection to 256, **2-layer BiLSTM**, attention pooling (one 256-d vector per clip) | same with a 1-layer BiLSTM over tokens | emotion is spread over the utterance (pitch contour, energy, rhythm); attention pooling can weight the emotionally loaded part (the target word) more than "Say the word" |
| Fusion | **gated**: a softmax over two weights per clip mixes the speech and text vectors, so the model's reliance on each modality can be read off; **concat**: concatenation + linear layer | | the gate makes the fusion inspectable; concat is the standard comparison |
| Classifier | MLP 256 -> 128 -> 7 with dropout 0.3 | | |

Training: AdamW (lr 1e-3, weight decay 1e-2), batch 32, cross-entropy, gradient clipping at 1.0,
up to 40 epochs; the epoch with the best validation macro-F1 is kept (patience 8).
Training is deterministic: seed 0 of the repeated speaker runs reproduces the first run exactly.

**Per-speaker normalisation** (`--speaker_norm`): an addition aimed at the unseen-speaker gap. Each speech
feature is z-scored using the mean and standard deviation of all frames from the same speaker. It uses no
labels, but it does use the unlabelled audio of the test speaker (in practice, some recordings of a new
speaker are needed before predicting).

## Results

Full tables are in `Results/`; accuracy in %, mean ± standard deviation over runs (n).

| Model | Random split | Word-disjoint | Unseen speaker |
|---|---|---|---|
| Speech (WavLM + BiLSTM) | 99.9 ± 0.1 (3) | 99.9 ± 0.1 (3) | 61.9 ± 7.9 (10) |
| Speech (log-mel + BiLSTM, baseline) | 99.5 ± 0.0 (3) | 99.6 ± 0.2 (3) | 41.9 ± 11.2 (10) |
| Speech (WavLM) + speaker norm. | - | - | 62.5 ± 5.6 (10) |
| Text (BERT + BiLSTM) | 12.0 ± 2.2 (3) | 14.3 ± 0.0 (3) | 14.3 ± 0.0 (2) |
| Fusion (gated) | 100.0 ± 0.0 (3) | 99.9 ± 0.1 (3) | 64.5 ± 7.2 (10) |
| Fusion (concat) | 99.8 ± 0.2 (3) | 99.4 ± 0.4 (3) | 59.3 ± 7.9 (10) |
| Fusion (gated) + speaker norm. | - | - | 58.3 ± 4.9 (10) |

Unseen speaker, same split and seed compared (`Results/paired_seed_comparison.md`):

| Comparison | Accuracy difference (points) | Better in |
|---|---|---|
| WavLM vs log-mel | +19.9 ± 8.2 | 10 / 10 |
| gated fusion vs speech only | +2.6 ± 8.7 | 6 / 10 |
| concat fusion vs speech only | -2.5 ± 6.7 | 4 / 10 |
| speaker normalisation (speech model) | +0.6 ± 6.6 | 7 / 10 |
| speaker normalisation (gated fusion) | -6.2 ± 9.3 | 2 / 10 |

What the numbers say:

- **Random and word-disjoint splits are solved by speech** (99.4-100%), so they cannot rank the models.
- **The unseen speaker is the real test.** Accuracy drops to about 62%, and depends a lot on the training
  seed (range 46-73% for the same model) and on the direction (training on the older voice and testing on
  the younger one is easier: 68.2 vs 55.5%).
- **Pretrained features matter most.** WavLM beats log-mel by 20 points on every one of the 10 pairs.
- **Text adds nothing on TESS.** The gated fusion puts weight 1.000 on speech for every test clip on every
  split. The fusion's speaker-split differences from speech-only (from -12 to +16 points) come from
  training a fresh speech branch, not from the text. Both fusions tie with speech-only within seed noise.
- **Per-speaker normalisation narrows the spread** (std 7.9 -> 5.6) and helps the harder direction
  (YAF -> OAF: 55.5 -> 58.6%), but the average gain (+0.6) is within noise.
- **Hardest emotions on an unseen speaker: happy** (F1 0.08) and pleasant surprise / angry (0.32);
  the easiest are neutral (0.88), sad (0.83) and fear (0.78). The new speaker's happy clips are confidently
  classified as angry: both are loud, high-pitched and fast, and the two actresses express them differently.
- In the t-SNE plots (`Results/plots/tsne_*.png`) the speech and fusion representations form seven clean
  clusters (silhouette 0.97 on the word split), the text representation forms none (silhouette -0.02),
  and on the unseen speaker the happy and angry clusters merge.

## Layout

```
common/
  data.py        manifest (de-duplicated) and the 8 splits
  prepare.py     CLI for data.py
  features.py    preprocessing + frozen WavLM / log-mel / BERT features, cached once
  modules.py     the model blocks (layer weights, BiLSTM encoder, attention pooling, fusion, classifier)
  runner.py      feature loading (+ speaker normalisation), training, testing, CLI used by models/
models/
  speech_pipeline/{train,test}.py   speech-only model
  text_pipeline/{train,test}.py     text-only model
  fusion_pipeline/{train,test}.py   speech + text fusion
scripts/
  audit_tess.py            dataset audit
  run_all.py               every variant on every split in one process (plans "main" and "speaker_seeds")
  aggregate.py             runs -> Results/ (tables, plots, failure cases, paired comparisons)
  make_kaggle_notebook.py  writes notebooks/emotion_kaggle.ipynb
notebooks/emotion_kaggle.ipynb   the notebook that produced runs/ on a Kaggle T4 GPU
data/            manifest.csv, splits.csv, durations.csv, tess_audit.json
runs/            one folder per run: config.json, history.csv, metrics.json, predictions.csv
Results/         accuracy tables (all variants), per-emotion F1, paired comparisons, failure cases, plots/
```

`runs/` leaves out each run's `model.pt` and `embeddings.npz` (about 700 MB together); they are
regenerated by the commands below.

## How to run

Install the requirements (`pip install -r ../requirements.txt`), download TESS
(Kaggle dataset `ejlok1/toronto-emotional-speech-set-tess`), then from this folder:

```bash
# 1. manifest and splits
python -m common.prepare --tess_root <path to TESS> --data_dir data
# 2. preprocessing and features (GPU recommended; about 2.5 min on a T4; the WavLM cache is about 4 GB)
python -m common.features --data_dir data --cache_dir cache
# 3a. one model, through the required pipeline scripts
python models/speech_pipeline/train.py --run_dir runs/speech_wavlm__word_s0 --split word_s0 --features wavlm
python models/speech_pipeline/test.py  --run_dir runs/speech_wavlm__word_s0
python models/text_pipeline/train.py   --run_dir runs/text_bert__word_s0 --split word_s0
python models/fusion_pipeline/train.py --run_dir runs/fusion_gated__word_s0 --split word_s0 --fusion gated
#    (each test.py takes the same --run_dir as its train.py; name run folders <variant>__<split>,
#     because aggregate.py reads the variant and split from the folder name)
# 3b. or every run at once: 40 runs (about 37 min on a T4) + 60 seed runs (about 52 min)
python scripts/run_all.py --data_dir data --cache_dir cache --runs_dir runs --plan main
python scripts/run_all.py --data_dir data --cache_dir cache --runs_dir runs --plan speaker_seeds
# 4. tables and plots
python scripts/aggregate.py --runs_dir runs --data_dir data --out Results
```

`aggregate.py` reproduces every table in `Results/` from the files in `runs/` as committed. The gate
histogram and the t-SNE plots also need the `embeddings.npz` files, so they are only redrawn after
step 3. On Kaggle, `notebooks/emotion_kaggle.ipynb` runs steps 1-3b (attach TESS and this folder as a
dataset, turn on GPU and internet).
