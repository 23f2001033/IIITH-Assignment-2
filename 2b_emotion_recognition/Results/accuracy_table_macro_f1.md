| Model | Random split (n) | Word-disjoint (n) | Speaker-independent (n) |
|---|---|---|---|
| Speech (WavLM + BiLSTM) | 99.9 ± 0.1 (3) | 99.9 ± 0.1 (3) | 55.7 ± 9.6 (10) |
| Speech (log-mel + BiLSTM, baseline) | 99.5 ± 0.0 (3) | 99.6 ± 0.2 (3) | 37.7 ± 13.9 (10) |
| Speech (WavLM + BiLSTM) + speaker norm. | - | - | 56.6 ± 8.1 (10) |
| Text (BERT + BiLSTM) | 4.6 ± 1.3 (3) | 5.6 ± 1.4 (3) | 7.1 ± 0.8 (2) |
| Fusion (gated) | 100.0 ± 0.0 (3) | 99.9 ± 0.1 (3) | 59.6 ± 8.1 (10) |
| Fusion (concat) | 99.8 ± 0.2 (3) | 99.4 ± 0.4 (3) | 53.8 ± 8.7 (10) |
| Fusion (gated) + speaker norm. | - | - | 51.9 ± 5.6 (10) |
