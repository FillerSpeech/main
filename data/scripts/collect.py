import argparse
import gzip
import json
import re
import sys
import subprocess
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
import spacy
from tqdm import tqdm

filler_list = ["ah", "aha", "eh", "ha", "hm", "huh", "oh", "uh", "um", "yeah", "ya"]
filler_list2 = ["well"]
model_name = "en_core_web_lg"


def init_worker(model_name, filler_list):
    global nlp, pattern
    try:
        nlp = spacy.load(model_name)
    except OSError:
        subprocess.run([sys.executable, "-m", "spacy", "download", model_name], check=True)
        nlp = spacy.load(model_name)
    import re
    pattern = re.compile(r"\b(" + "|".join(map(re.escape, filler_list)) + r")\b", flags=re.IGNORECASE)


def highlight_and_flag(text: str) -> tuple[str, bool]:
    global nlp, pattern
    doc = nlp(text)
    parts = []
    for token in doc:
        if token.lemma_.lower() == "well" and token.pos_ == "INTJ":
            parts.append(f"<{token.text}>")
        else:
            parts.append(token.text)
        parts.append(token.whitespace_)
    text_stage1 = "".join(parts)

    text_final, count = pattern.subn(lambda m: f"<{m.group(0)}>", text_stage1)
    return text_final, (count > 0)


def process_data(data):
    text = data['supervisions'][0]['text']
    new_text, flag = highlight_and_flag(text)
    if flag:
        data['supervisions'][0]['text_filler'] = new_text
        return data
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Path to input JSONL.GZ manifest")
    parser.add_argument("--subset", type=str, required=True,
                        help="Subset name for logging and output naming")
    parser.add_argument("--output_dir", type=Path, default=Path('./data/libriheavy/temp'),
                        help="Directory to save the output JSONL.GZ files")
    parser.add_argument("--num-workers", type=int, default=cpu_count(),
                        help="Number of worker processes (default: CPU cores)")
    args = parser.parse_args()

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    output_path = out_dir / f"filler_inclusive_{args.subset}.jsonl.gz"

    with gzip.open(args.manifest, 'rt', encoding='utf-8') as f_in, \
         gzip.open(output_path, 'wt', encoding='utf-8') as f_out, \
         Pool(processes=args.num_workers, initializer=init_worker, initargs=(model_name, filler_list)) as pool:

        data_gen = (json.loads(line) for line in f_in)
        chunksize = 100
        for rec in tqdm(pool.imap_unordered(process_data, data_gen, chunksize),
                        desc=f"Collecting subset {args.subset}", unit="rec"):
            if rec:
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

if __name__ == '__main__':
    main()