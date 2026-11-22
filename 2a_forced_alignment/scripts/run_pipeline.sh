#!/usr/bin/env bash
# Full forced-alignment pipeline: raw data -> corpus -> dictionaries -> four alignment runs.
# Run from 2a_forced_alignment/. Needs MFA 3.x on PATH (or MFA="micromamba run -n mfa mfa").
#
#   A_arpa      english_us_arpa acoustic model + ARPAbet dictionary + G2P pronunciations for OOV words
#   B_mfa       english_mfa acoustic model + MFA (IPA-like) dictionary + G2P pronunciations for OOV words
#   C_adapted   english_us_arpa adapted to these two speakers with `mfa adapt`, same dictionary as A
#   D_no_g2p    english_us_arpa + the base dictionary only: OOV words fall back to <unk>/spn
#   A_default_beam  run A with MFA's default beam, to show whether the long passages fail without a wider beam
set -euo pipefail

MFA=${MFA:-mfa}
PY=${PY:-python}
export MFA_ROOT_DIR=${MFA_ROOT_DIR:-$HOME/mfa_root}
WORK=${WORK:-/tmp/mfa_work}
CORPUS=data/corpus
OUT=outputs
DICT_DIR=dictionaries
BEAM="--beam 100 --retry_beam 400"
COMMON="--clean --num_jobs 4 --output_format long_textgrid"
PRETRAINED=$MFA_ROOT_DIR/pretrained_models

mkdir -p "$OUT/textgrids" "$OUT/logs" "$DICT_DIR" "$WORK"
$MFA version | tee "$OUT/logs/mfa_version.txt"

echo "== 1. Prepare corpus"
$PY scripts/prepare_corpus.py --raw data/raw --out "$CORPUS"

echo "== 2. Download pretrained models"
for kind_name in "acoustic english_us_arpa" "dictionary english_us_arpa" "g2p english_us_arpa" \
                 "acoustic english_mfa" "dictionary english_us_mfa" "g2p english_us_mfa"; do
  $MFA model download $kind_name --ignore_cache || $MFA model download $kind_name
done
ARPA_DICT=$PRETRAINED/dictionary/english_us_arpa.dict
MFA_DICT=$PRETRAINED/dictionary/english_us_mfa.dict

echo "== 3. Validate the corpus (reports OOV words and audio problems)"
$MFA validate "$CORPUS" english_us_arpa --acoustic_model_path english_us_arpa --clean \
  --temporary_directory "$WORK/validate" 2>&1 | tee "$OUT/logs/validate.log" || true
cp "$WORK"/validate/*/oovs_found_*.txt "$WORK"/validate/*/utterance_oovs.txt "$OUT/logs/" 2>/dev/null || true
cp "$WORK"/validate/*.log "$WORK"/validate/*/*.log "$OUT/logs/" 2>/dev/null || true

echo "== 4. Custom dictionaries: G2P pronunciations for words missing from each base dictionary"
for set in arpa mfa; do
  base=$([ $set = arpa ] && echo "$ARPA_DICT" || echo "$MFA_DICT")
  g2p=$([ $set = arpa ] && echo english_us_arpa || echo english_us_mfa)
  $PY scripts/find_oovs.py --corpus "$CORPUS" --dictionary "$base" --out "$DICT_DIR/oovs_$set.txt"
  $MFA g2p "$DICT_DIR/oovs_$set.txt" $g2p "$DICT_DIR/g2p_$set.dict" --num_pronunciations 1 --clean \
    --temporary_directory "$WORK/g2p_$set"
  # G2P output is checked by hand; wrong entries are replaced from dictionaries/manual_$set.txt
  $PY scripts/apply_overrides.py --base "$base" --g2p "$DICT_DIR/g2p_$set.dict"     --overrides "$DICT_DIR/manual_$set.txt" --out "$WORK/custom_$set.dict" --report "$DICT_DIR/oov_final_$set.dict"
done

run_align() {  # name dictionary acoustic_model extra_args...
  local name=$1 dict=$2 model=$3; shift 3
  echo "== align $name"
  local start=$(date +%s)
  $MFA align "$CORPUS" "$dict" "$model" "$OUT/textgrids/$name" $COMMON "$@" \
    --temporary_directory "$WORK/$name" > "$OUT/logs/align_$name.log" 2>&1 || echo "align $name FAILED (see log)"
  echo "$name $(( $(date +%s) - start ))s" >> "$OUT/logs/timings.txt"
  tail -5 "$OUT/logs/align_$name.log"
}

echo "== 5. Alignment runs"
run_align A_arpa "$WORK/custom_arpa.dict" english_us_arpa $BEAM
run_align B_mfa "$WORK/custom_mfa.dict" english_mfa $BEAM
run_align D_no_g2p english_us_arpa english_us_arpa $BEAM
run_align A_default_beam "$WORK/custom_arpa.dict" english_us_arpa

echo "== 6. Speaker adaptation (C)"
# `mfa adapt --output_directory` exports the alignment made BEFORE adaptation ("Alignment already done,
# skipping"), so the adapted model is saved and the corpus is aligned again with it.
$MFA adapt "$CORPUS" "$WORK/custom_arpa.dict" english_us_arpa "$WORK/adapted_arpa.zip" --clean $BEAM   --temporary_directory "$WORK/adapt" > "$OUT/logs/adapt.log" 2>&1 || echo "adapt FAILED (see log)"
tail -3 "$OUT/logs/adapt.log"
run_align C_adapted "$WORK/custom_arpa.dict" "$WORK/adapted_arpa.zip" $BEAM

echo "== Done. TextGrids per run:"
for d in "$OUT"/textgrids/*/; do echo "$d $(find "$d" -name '*.TextGrid' | wc -l)"; done
