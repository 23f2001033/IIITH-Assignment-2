"""Speech + text (multimodal) emotion recognition. (train)

Blocks: the speech branch (see speech_pipeline) and the text branch (see text_pipeline), trained jointly -> fusion (--fusion gated: per-clip softmax
weights over the two 256-d representations; --fusion concat: concatenation + linear layer) ->
classifier (MLP, 7 emotions).

Train on the split's train ids, early-stop on validation macro-F1, save model.pt to --run_dir.
Run from 2b_emotion_recognition/ after `python -m common.prepare` and `python -m common.features`:
    python models/fusion_pipeline/train.py --run_dir runs/fusion_word_s0 --split word_s0 --fusion gated
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from common.runner import cli  # noqa: E402

if __name__ == "__main__":
    cli("fusion", "train")
