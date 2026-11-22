"""Text-only emotion recognition. (train)

Blocks: preprocessing (lower-case, WordPiece tokens of "say the word <word>") -> feature extraction (frozen
BERT-base-uncased, learned weighted sum of its 13 layers) -> contextual modelling (BiLSTM + attention
pooling over tokens) -> classifier (MLP, 7 emotions).

Train on the split's train ids, early-stop on validation macro-F1, save model.pt to --run_dir.
Run from 2b_emotion_recognition/ after `python -m common.prepare` and `python -m common.features`:
    python models/text_pipeline/train.py --run_dir runs/text_word_s0 --split word_s0
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from common.runner import cli  # noqa: E402

if __name__ == "__main__":
    cli("text", "train")
