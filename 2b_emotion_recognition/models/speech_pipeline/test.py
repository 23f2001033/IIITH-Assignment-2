"""Speech-only emotion recognition. (test)

Blocks: preprocessing (16 kHz, silence trim, peak norm) -> feature extraction (frozen WavLM Base+, learned
weighted sum of its 13 layers; or 80-band log-mel with --features logmel) -> temporal modelling
(2-layer BiLSTM + attention pooling) -> classifier (MLP, 7 emotions).

Evaluate a trained run on its split's test ids: metrics.json, predictions.csv, embeddings.npz.
Run from 2b_emotion_recognition/ after `python -m common.prepare` and `python -m common.features`:
    python models/speech_pipeline/test.py --run_dir runs/speech_word_s0
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from common.runner import cli  # noqa: E402

if __name__ == "__main__":
    cli("speech", "test")
