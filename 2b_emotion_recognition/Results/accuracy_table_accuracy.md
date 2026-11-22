| Model | Random split (n) | Word-disjoint (n) | Speaker-independent (n) |
|---|---|---|---|
| Speech (WavLM + BiLSTM) | 99.9 ± 0.1 (3) | 99.9 ± 0.1 (3) | 61.9 ± 7.9 (10) |
| Speech (log-mel + BiLSTM, baseline) | 99.5 ± 0.0 (3) | 99.6 ± 0.2 (3) | 41.9 ± 11.2 (10) |
| Speech (WavLM + BiLSTM) + speaker norm. | - | - | 62.5 ± 5.6 (10) |
| Text (BERT + BiLSTM) | 12.0 ± 2.2 (3) | 14.3 ± 0.0 (3) | 14.3 ± 0.0 (2) |
| Fusion (gated) | 100.0 ± 0.0 (3) | 99.9 ± 0.1 (3) | 64.5 ± 7.2 (10) |
| Fusion (concat) | 99.8 ± 0.2 (3) | 99.4 ± 0.4 (3) | 59.3 ± 7.9 (10) |
| Fusion (gated) + speaker norm. | - | - | 58.3 ± 4.9 (10) |
