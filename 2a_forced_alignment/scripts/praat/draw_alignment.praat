# Draw a spectrogram, the waveform and every tier of a TextGrid for one time window, and save a PNG.
# Run headless:  Praat.exe --run draw_alignment.praat <wav> <TextGrid> <start> <end> <png> <title>
# start = end = 0 draws the whole file.
form Draw alignment
    sentence wav_path
    sentence textgrid_path
    real start 0
    real end 0
    sentence png_path
    sentence title
endform

sound = Read from file: wav_path$
grid = Read from file: textgrid_path$
if start = 0 and end = 0
    selectObject: sound
    end = Get end time
endif
selectObject: grid
n_tiers = Get number of tiers

selectObject: sound
part = Extract part: start, end, "rectangular", 1, "yes"
spec = To Spectrogram: 0.005, 5000, 0.002, 20, "Gaussian"

Erase all
Font size: 10
Line width: 1
width = 11
spec_bottom = 2.4
tier_bottom = spec_bottom + 1.2 + 0.45 * n_tiers

# Spectrogram on top
Select outer viewport: 0, width, 0, spec_bottom + 0.3
selectObject: spec
Paint: start, end, 0, 5000, 100, "yes", 50, 6, 0, "no"
Draw inner box
Marks left every: 1, 1000, "yes", "yes", "no"
Text left: "yes", "Hz"
Text top: "yes", title$

# Waveform with all TextGrid tiers below it, on the same time axis
Select outer viewport: 0, width, spec_bottom, tier_bottom
selectObject: sound, grid
Draw: start, end, "yes", "yes", "yes"

Select outer viewport: 0, width, 0, tier_bottom
Save as 300-dpi PNG file: png_path$
