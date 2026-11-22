"""Model blocks shared by the three pipelines.

Feature extraction (learned part): a softmax-weighted sum over the 13 layers of the frozen encoder,
then LayerNorm and a linear projection to 256 dimensions.
Temporal / contextual modelling: a bidirectional LSTM over time steps (speech) or tokens (text),
followed by attention pooling, giving one 256-d vector per clip. This vector is the representation
used for the cluster plots.
Fusion: either a modality gate (two softmax weights per clip, so the model's reliance on each modality
can be read off) or concatenation followed by a linear layer.
Classifier: two-layer MLP with dropout to 7 emotions.
"""
import torch
import torch.nn as nn


class LayerWeights(nn.Module):
    """Learned softmax weights over encoder layers: [B, n_layers, T, D] -> [B, T, D]."""

    def __init__(self, n_layers):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_layers))

    def forward(self, x):
        w = torch.softmax(self.logits, dim=0)
        return (x * w[None, :, None, None]).sum(dim=1)


class AttentionPool(nn.Module):
    """Additive attention over valid steps; returns the pooled vector and the attention weights."""

    def __init__(self, dim, hidden=128):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, h, mask):
        scores = self.score(h).squeeze(-1).masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        return (h * weights.unsqueeze(-1)).sum(dim=1), weights


class SequenceEncoder(nn.Module):
    """Feature projection + BiLSTM + attention pooling -> [B, 2*hidden]."""

    def __init__(self, in_dim, n_layers=None, proj=256, hidden=128, lstm_layers=2, dropout=0.3):
        super().__init__()
        self.layer_weights = LayerWeights(n_layers) if n_layers else None
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Sequential(nn.Linear(in_dim, proj), nn.GELU(), nn.Dropout(dropout))
        self.lstm = nn.LSTM(proj, hidden, num_layers=lstm_layers, batch_first=True, bidirectional=True,
                            dropout=dropout if lstm_layers > 1 else 0.0)
        self.pool = AttentionPool(2 * hidden)
        self.out_dim = 2 * hidden

    def forward(self, x, mask):
        if self.layer_weights is not None:
            x = self.layer_weights(x)
        x = self.proj(self.norm(x))
        lengths = mask.sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        h, _ = self.lstm(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=x.shape[1])
        pooled, attention = self.pool(h, mask)
        return pooled, attention


class Classifier(nn.Module):
    def __init__(self, in_dim, n_classes=7, hidden=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, x):
        return self.net(x)


class UnimodalModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.classifier = Classifier(encoder.out_dim)

    def forward(self, batch):
        z, _ = self.encoder(batch["x"], batch["mask"])
        return {"logits": self.classifier(z), "embedding": z}


class FusionModel(nn.Module):
    def __init__(self, speech_encoder, text_encoder, mode="gated"):
        super().__init__()
        self.speech, self.text, self.mode = speech_encoder, text_encoder, mode
        dim = speech_encoder.out_dim
        assert text_encoder.out_dim == dim
        if mode == "gated":
            self.gate = nn.Linear(2 * dim, 2)
        else:
            self.mix = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU())
        self.classifier = Classifier(dim)

    def forward(self, batch):
        hs, _ = self.speech(batch["x_speech"], batch["mask_speech"])
        ht, _ = self.text(batch["x_text"], batch["mask_text"])
        both = torch.cat([hs, ht], dim=-1)
        out = {"speech_embedding": hs, "text_embedding": ht}
        if self.mode == "gated":
            gate = torch.softmax(self.gate(both), dim=-1)
            z = gate[:, :1] * hs + gate[:, 1:] * ht
            out["gate"] = gate
        else:
            z = self.mix(both)
        out.update(logits=self.classifier(z), embedding=z)
        return out
