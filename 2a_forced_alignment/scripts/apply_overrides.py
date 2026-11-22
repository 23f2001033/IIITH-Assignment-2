"""Merge a base dictionary, G2P pronunciations for OOV words, and hand-checked corrections.

G2P is a statistical guess, so its output is reviewed by hand. Wrong entries are replaced by the
lines in the overrides file. An override is a word followed by phones, where a token starting
with @ is replaced by the first pronunciation of that word in the base dictionary, for example
`depoliticize  D IY0 @politicize`. Every phone of the result must exist in the base dictionary,
so a typo cannot reach the aligner.

usage: python scripts/apply_overrides.py --base english_us_arpa.dict --g2p g2p_arpa.dict
           --overrides dictionaries/manual_arpa.txt --out custom_arpa.dict --report dictionaries/oov_final_arpa.dict
"""
import argparse


def read_dict(path):
    entries = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            # MFA dictionaries may carry probability columns before the phones; keep only phones.
            phones = [p for p in parts[1:] if not _is_number(p)]
            entries.setdefault(parts[0].lower(), []).append(phones)
    return entries


def _is_number(token):
    try:
        float(token)
        return True
    except ValueError:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--g2p", required=True)
    parser.add_argument("--overrides", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    base = read_dict(args.base)
    inventory = {p for prons in base.values() for pron in prons for p in pron}
    final = {w: prons[0] for w, prons in read_dict(args.g2p).items()}
    source = {w: "g2p" for w in final}

    with open(args.overrides, encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            word, *tokens = line.split()
            phones = []
            for token in tokens:
                if token.startswith("@"):
                    ref = token[1:].lower()
                    if ref not in base:
                        phones = None
                        print("WARNING override for {}: reference word '{}' not in base dictionary; "
                              "keeping the G2P pronunciation".format(word, ref))
                        break
                    phones += base[ref][0]
                else:
                    phones.append(token)
            if phones is None:
                continue
            unknown = [p for p in phones if p not in inventory]
            if unknown:
                print("WARNING override for {}: phones {} are not in the acoustic model's inventory; "
                      "keeping the G2P pronunciation".format(word, unknown))
                continue
            if word.lower() in final:
                print("override {:15s} g2p: {:30s} -> {}".format(word, " ".join(final[word.lower()]), " ".join(phones)))
            final[word.lower()] = phones
            source[word.lower()] = "manual"

    with open(args.report, "w", encoding="utf-8", newline="\n") as f:
        for word in sorted(final):
            f.write("{}\t{}\t# {}\n".format(word, " ".join(final[word]), source[word]))
    with open(args.base, encoding="utf-8") as src, open(args.out, "w", encoding="utf-8", newline="\n") as out:
        out.write(src.read())
        for word in sorted(final):
            out.write("{}\t{}\n".format(word, " ".join(final[word])))


if __name__ == "__main__":
    main()
