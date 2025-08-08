import argparse
import gzip
import json
import subprocess
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
import os
import logging

from nemo_text_processing.text_normalization.normalize import Normalizer
from text_normalizer import EnglishTextNormalizer

import re
from concurrent.futures import ProcessPoolExecutor

whisper_normalizer = None
nemo = None
OUTPUT_DIR = None

def quiet_nemo_logging():
    os.environ.setdefault("NEMO_LOG_LEVEL", "ERROR")
    for name in ("nemo_text_processing", "NeMo-text-processing", "NeMo"):
        logging.getLogger(name).setLevel(logging.ERROR)


def init_worker(output_dir: str):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NEMO_LOG_LEVEL", "ERROR")

    import logging
    for name in ("nemo_text_processing", "NeMo-text-processing", "NeMo"):
        logging.getLogger(name).setLevel(logging.ERROR)

    global whisper_normalizer, nemo, OUTPUT_DIR
    whisper_normalizer = EnglishTextNormalizer()
    nemo = Normalizer(lang="en", input_case="cased")
    OUTPUT_DIR = output_dir



def capitalize_after_punctuation(match):
    return match.group(1) + match.group(2).upper()
def revise(text):
    text = text.capitalize()
    text = text.replace(' ?', '?').replace(' .', '.').replace(' ,', ',').replace('  ',' ').replace(' !', '!')
    text = re.sub(r'^[\.,\s]+', '', text)
    text = re.sub(r'([.,])\1+', r'\1', text)
    text = text.replace('.,' ,'.').replace(',?', '?').replace(',.', '.')
    text = text.replace(' i ', " I ").replace("i'm", "I'm").replace("i'll", "I'll")
    text = text.replace('"i ', '"I ').replace("i'd", "I'd").replace("'i ", "'I")
    text = re.sub(r'([.?!]\s+)(\w)', capitalize_after_punctuation, text)
    return text

def process(record):
    quiet_nemo_logging()

    text = record['supervisions'][0]['text'].replace('(', '').replace(')', '')
    norm_text1 = whisper_normalizer(text)
    norm_text2 = nemo.normalize(norm_text1, punct_post_process=True)

    part_subset, spk, _, basename = record['id'].split('/')
    id_ = "/".join([part_subset, spk, basename])

    lab_path = Path(OUTPUT_DIR) / f"audio/{id_}.lab"
    txt_path = Path(OUTPUT_DIR) / f"text/{id_}.txt"
    lab_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lab_path, 'w') as f:
        f.write(norm_text2)
    with open(txt_path, 'w') as f:
        f.write(norm_text2)


def main():
    quiet_nemo_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument('--jsonl-gz', required=True, help="Path to the JSONL.GZ file")
    parser.add_argument('--subset', required=True, help="Subset name")
    parser.add_argument('--output-dir', required=True, help="Directory where normalized text files will be saved")
    parser.add_argument('--num-workers', type=int, default=os.cpu_count() // 2, help="Number of parallel workers for segmentation")
    args = parser.parse_args()


    records = []
    with gzip.open(args.jsonl_gz, 'rt', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))
    chunksize = max(1, len(records) // ((args.num_workers or 1) * 4))

    print(f"Loaded {len(records)} cuts; normalizing text with {args.num_workers} workers...")


    with ProcessPoolExecutor(max_workers=args.num_workers,
                             initializer=init_worker, initargs=(args.output_dir,)) as exe:
        for _ in tqdm(
            exe.map(process, records, chunksize=chunksize),
            total=len(records),
            desc=f"Normalizing batches of {args.subset}"
        ):
            pass

if __name__ == '__main__':
    main()
