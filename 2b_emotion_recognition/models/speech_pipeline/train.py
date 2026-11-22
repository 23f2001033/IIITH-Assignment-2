"""Speech-only emotion recognition. (train)

Blocks: preprocessing (16 kHz, silence trim, peak norm) -> feature extraction (frozen WavLM Base+, learned
weighted sum of its 13 layers; or 80-band log-mel with --features logmel) -> temporal modelling
(2-layer BiLSTM + attention pooling) -> classifier (MLP, 7 emotions).

Train on the split's train ids, early-stop on validation macro-F1, save model.pt to --run_dir.
Run from 2b_emotion_recognition/ after `python -m common.prepare` and `python -m common.features`:
    python models/speech_pipeline/train.py --run_dir runs/speech_word_s0 --split word_s0 --features wavlm
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from common.runner import cli  # noqa: E402

if __name__ == "__main__":
    cli("speech", "train")
