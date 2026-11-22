# Measure F1 and F2 at 25%, 50% and 75% of each vowel interval listed in a table, plus mean F0, and
# print the result as CSV to standard output (printing needs no --FULL-TRUST, unlike writing files).
# Run headless:  Praat.exe --run measure_formants.praat <wav> <intervals.tsv> <max_formant_hz>
# intervals.tsv has a header and rows: label<TAB>start<TAB>end
# max_formant_hz is the formant ceiling: about 5000 for an adult male voice, 5500 for a female voice.
form Measure formants
    sentence wav_path
    sentence intervals_path
    real max_formant 5000
endform

sound = Read from file: wav_path$
formant = To Formant (burg): 0, 5, max_formant, 0.025, 50
selectObject: sound
pitch = To Pitch: 0, 75, 500
table = Read Table from tab-separated file: intervals_path$
n = Get number of rows

result = Create Table with column names: "result", n, "label start end duration_ms f0 f1_25 f2_25 f1_50 f2_50 f1_75 f2_75"
for i to n
    selectObject: table
    label$ = Get value: i, "label"
    t1 = Get value: i, "start"
    t2 = Get value: i, "end"
    selectObject: pitch
    f0 = Get mean: t1, t2, "Hertz"
    selectObject: result
    Set string value: i, "label", label$
    Set numeric value: i, "start", t1
    Set numeric value: i, "end", t2
    Set numeric value: i, "duration_ms", round((t2 - t1) * 1000)
    Set numeric value: i, "f0", round(f0)
    for p from 1 to 3
        t = t1 + (t2 - t1) * p / 4
        selectObject: formant
        f1 = Get value at time: 1, t, "hertz", "linear"
        f2 = Get value at time: 2, t, "hertz", "linear"
        selectObject: result
        Set numeric value: i, "f1_" + string$(25 * p), round(f1)
        Set numeric value: i, "f2_" + string$(25 * p), round(f2)
    endfor
endfor
writeInfoLine: "label,start,end,duration_ms,f0,f1_25,f2_25,f1_50,f2_50,f1_75,f2_75"
for i to n
    selectObject: result
    line$ = Get value: i, "label"
    for col from 2 to 11
        column$ = Get column label: col
        value$ = Get value: i, column$
        line$ = line$ + "," + value$
    endfor
    appendInfoLine: line$
endfor
