# IIIT Hyderabad speech lab, Assignment 2

| Part | Folder | Summary |
|---|---|---|
| 2a | [`2a_forced_alignment/`](2a_forced_alignment/README.md) | Word and phone forced alignment of 6 recordings with the Montreal Forced Aligner: two pretrained acoustic models, a speaker-adapted model and a custom G2P dictionary, inspected in Praat and compared across models, with the learner's vowels measured inside the aligned intervals |
| 2b | [`2b_emotion_recognition/`](2b_emotion_recognition/README.md) | Speech-only, text-only and fusion emotion recognition on TESS, evaluated on random, unseen-word and unseen-speaker splits |

Each folder's README gives the install, data preparation and run commands.
Python packages for both parts are listed in `requirements.txt`; MFA itself is installed with conda (see 2a).
The report covering both parts is `report/Assignment2_Report.pdf` (source: `report/report.html`).
