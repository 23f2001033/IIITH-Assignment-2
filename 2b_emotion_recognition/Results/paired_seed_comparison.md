Speaker-independent splits, A minus B on the same split and seed (percentage points).

| Comparison | Metric | Mean diff ± std | A better in | Range |
|---|---|---|---|---|
| pretrained WavLM vs hand-crafted log-mel | accuracy | +19.9 ± 8.2 | 10/10 | +9.4 to +36.1 |
| pretrained WavLM vs hand-crafted log-mel | macro_f1 | +18.0 ± 9.3 | 10/10 | +7.4 to +36.4 |
| gated fusion vs speech only | accuracy | +2.6 ± 8.7 | 6/10 | -12.4 to +16.3 |
| gated fusion vs speech only | macro_f1 | +3.8 ± 9.4 | 6/10 | -12.7 to +16.6 |
| concat fusion vs speech only | accuracy | -2.5 ± 6.7 | 4/10 | -12.4 to +8.1 |
| concat fusion vs speech only | macro_f1 | -1.9 ± 6.6 | 3/10 | -13.8 to +9.4 |
| speaker normalisation, speech model | accuracy | +0.6 ± 6.6 | 7/10 | -11.6 to +13.8 |
| speaker normalisation, speech model | macro_f1 | +0.9 ± 6.9 | 6/10 | -11.2 to +12.5 |
| speaker normalisation, gated fusion | accuracy | -6.2 ± 9.3 | 2/10 | -23.5 to +8.3 |
| speaker normalisation, gated fusion | macro_f1 | -7.7 ± 9.1 | 2/10 | -22.9 to +6.8 |
