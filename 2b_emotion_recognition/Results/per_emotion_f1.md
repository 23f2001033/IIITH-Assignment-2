| Model | Split | angry | disgust | fear | happy | neutral | pleasant_surprise | sad |
|---|---|---|---|---|---|---|---|---|
| Speech (WavLM + BiLSTM) | random | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Speech (WavLM + BiLSTM) | word | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Speech (WavLM + BiLSTM) | speaker | 0.32 | 0.70 | 0.78 | 0.08 | 0.88 | 0.32 | 0.83 |
| Speech (log-mel + BiLSTM, baseline) | random | 1.00 | 0.99 | 1.00 | 0.99 | 1.00 | 0.99 | 1.00 |
| Speech (log-mel + BiLSTM, baseline) | word | 0.99 | 0.99 | 0.99 | 1.00 | 1.00 | 0.99 | 1.00 |
| Speech (log-mel + BiLSTM, baseline) | speaker | 0.37 | 0.41 | 0.57 | 0.07 | 0.76 | 0.07 | 0.39 |
| Speech (WavLM + BiLSTM) + speaker norm. | speaker | 0.10 | 0.72 | 0.76 | 0.14 | 0.91 | 0.44 | 0.91 |
| Text (BERT + BiLSTM) | random | 0.05 | 0.04 | 0.00 | 0.01 | 0.09 | 0.11 | 0.03 |
| Text (BERT + BiLSTM) | word | 0.03 | 0.00 | 0.14 | 0.14 | 0.00 | 0.08 | 0.00 |
| Text (BERT + BiLSTM) | speaker | 0.00 | 0.07 | 0.11 | 0.12 | 0.00 | 0.20 | 0.00 |
| Fusion (gated) | random | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Fusion (gated) | word | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Fusion (gated) | speaker | 0.46 | 0.67 | 0.84 | 0.12 | 0.84 | 0.43 | 0.81 |
| Fusion (concat) | random | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 | 1.00 |
| Fusion (concat) | word | 1.00 | 0.98 | 1.00 | 1.00 | 1.00 | 0.99 | 0.99 |
| Fusion (concat) | speaker | 0.12 | 0.66 | 0.73 | 0.28 | 0.85 | 0.39 | 0.74 |
| Fusion (gated) + speaker norm. | speaker | 0.14 | 0.70 | 0.74 | 0.07 | 0.87 | 0.32 | 0.79 |
