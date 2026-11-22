| Model | Train OAF → test YAF | Train YAF → test OAF |
|---|---|---|
| Speech (WavLM + BiLSTM) | 68.2 ± 3.2 [64.1-73.0], n=5 | 55.5 ± 5.8 [45.8-62.5], n=5 |
| Speech (log-mel + BiLSTM, baseline) | 51.5 ± 3.7 [46.4-56.3], n=5 | 32.3 ± 7.2 [18.1-38.0], n=5 |
| Speech (WavLM + BiLSTM) + speaker norm. | 66.4 ± 4.2 [58.6-70.4], n=5 | 58.6 ± 3.8 [54.8-65.3], n=5 |
| Text (BERT + BiLSTM) | 14.3 | 14.3 |
| Fusion (gated) | 68.5 ± 8.4 [55.6-80.5], n=5 | 60.5 ± 1.3 [58.1-62.1], n=5 |
| Fusion (concat) | 64.9 ± 7.9 [53.3-73.7], n=5 | 53.8 ± 1.9 [50.6-56.2], n=5 |
| Fusion (gated) + speaker norm. | 58.1 ± 4.4 [53.4-66.4], n=5 | 58.4 ± 5.3 [52.9-68.4], n=5 |
