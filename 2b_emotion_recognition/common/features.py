"""Preprocessing and feature extraction, run once and cached.

Speech preprocessing: mono, resample to 16 kHz (TESS is 24.4 kHz, one file is 96 kHz), trim leading and
trailing silence (top_db 30), peak-normalise. Variable length is kept; batches are padded with a mask.
Speech features (time_steps x features):
  wavlm   all 13 hidden layers of WavLM Base+ (frozen), 50 frames/s x 768, stored as float16
  logmel  80-band log-mel spectrogram, 25 ms window, 10 ms hop: a hand-crafted baseline
Text preprocessing: the transcript is lower-cased and tokenised with BERT's WordPiece tokenizer.
Text features (tokens x features): all 13 hidden layers of BERT-base-uncased (frozen).

usage: python -m common.features --data_dir data --cache_dir cache --device cuda
"""
import argparse
import os
import time

import librosa
import numpy as np
import soundfile as sf
import torch

from common.data import read_csv

SAMPLE_RATE = 16000
WAVLM = "microsoft/wavlm-base-plus"
BERT = "bert-base-uncased"


def load_audio(path):
    wav, rate = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if rate != SAMPLE_RATE:
        wav = librosa.resample(wav, orig_sr=rate, target_sr=SAMPLE_RATE)
    raw_seconds = len(wav) / SAMPLE_RATE
    wav, _ = librosa.effects.trim(wav, top_db=30)
    wav = wav / (np.abs(wav).max() + 1e-8) * 0.95
    return wav.astype(np.float32), raw_seconds


def log_mel(wav):
    mel = librosa.feature.melspectrogram(y=wav, sr=SAMPLE_RATE, n_fft=400, hop_length=160, n_mels=80)
    return np.log(mel + 1e-6).T.astype(np.float32)


@torch.no_grad()
def extract(data_dir, cache_dir, device):
    from transformers import AutoModel, AutoTokenizer, WavLMModel

    os.makedirs(cache_dir, exist_ok=True)
    rows = read_csv(os.path.join(data_dir, "manifest.csv"))

    wavlm = WavLMModel.from_pretrained(WAVLM).to(device).eval()
    speech_wavlm, speech_logmel, durations = {}, {}, []
    start = time.time()
    for i, r in enumerate(rows):
        wav, raw_seconds = load_audio(r["path"])
        durations.append({"id": r["id"], "raw_seconds": round(raw_seconds, 3),
                          "trimmed_seconds": round(len(wav) / SAMPLE_RATE, 3)})
        speech_logmel[r["id"]] = torch.from_numpy(log_mel(wav)).half()
        x = torch.from_numpy(wav)[None].to(device)
        x = (x - x.mean()) / (x.std() + 1e-7)  # WavLM Base+ expects zero-mean, unit-variance input
        hidden = wavlm(x, output_hidden_states=True).hidden_states
        speech_wavlm[r["id"]] = torch.stack([h[0] for h in hidden]).half().cpu()  # [13, T, 768]
        if i % 500 == 0:
            print("speech", i, len(rows), round(time.time() - start), "s", flush=True)
    del wavlm
    torch.save(speech_wavlm, os.path.join(cache_dir, "speech_wavlm.pt"))
    torch.save(speech_logmel, os.path.join(cache_dir, "speech_logmel.pt"))

    tokenizer = AutoTokenizer.from_pretrained(BERT)
    bert = AutoModel.from_pretrained(BERT).to(device).eval()
    text_bert, tokens = {}, {}
    for r in rows:
        enc = tokenizer(r["text"].lower(), return_tensors="pt").to(device)
        hidden = bert(**enc, output_hidden_states=True).hidden_states
        text_bert[r["id"]] = torch.stack([h[0] for h in hidden]).half().cpu()  # [13, L, 768]
        tokens[r["id"]] = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    torch.save(text_bert, os.path.join(cache_dir, "text_bert.pt"))
    torch.save(tokens, os.path.join(cache_dir, "text_tokens.pt"))

    with open(os.path.join(cache_dir, "durations.csv"), "w") as f:
        f.write("id,raw_seconds,trimmed_seconds\n")
        f.write("".join("{id},{raw_seconds},{trimmed_seconds}\n".format(**d) for d in durations))
    print("features done in", round(time.time() - start), "s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    extract(args.data_dir, args.cache_dir, args.device)


if __name__ == "__main__":
    main()
