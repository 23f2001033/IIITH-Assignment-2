"""Text-only emotion recognition. (test)

Blocks: preprocessing (lower-case, WordPiece tokens of "say the word <word>") -> feature extraction (frozen
BERT-base-uncased, learned weighted sum of its 13 layers) -> contextual modelling (BiLSTM + attention
pooling over tokens) -> classifier (MLP, 7 emotions).

Evaluate a trained run on its split's test ids: metrics.json, predictions.csv, embeddings.npz.
Run from 2b_emotion_recognition/ after `python -m common.prepare` and `python -m common.features`:
    python models/text_pipeline/test.py --run_dir runs/text_word_s0
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from common.runner import cli  # noqa: E402

if __name__ == "__main__":
    cli("text", "test")
