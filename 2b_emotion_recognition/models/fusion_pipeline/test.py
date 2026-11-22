"""Speech + text (multimodal) emotion recognition. (test)

Blocks: the speech branch (see speech_pipeline) and the text branch (see text_pipeline), trained jointly -> fusion (--fusion gated: per-clip softmax
weights over the two 256-d representations; --fusion concat: concatenation + linear layer) ->
classifier (MLP, 7 emotions).

Evaluate a trained run on its split's test ids: metrics.json, predictions.csv, embeddings.npz.
Run from 2b_emotion_recognition/ after `python -m common.prepare` and `python -m common.features`:
    python models/fusion_pipeline/test.py --run_dir runs/fusion_word_s0
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from common.runner import cli  # noqa: E402

if __name__ == "__main__":
    cli("fusion", "test")
