import argparse
import gzip
import json
import subprocess
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
import os

def segment_record(args):
    record, input_dir, output_dir = args
    rec = record.get('recording', {})
    src = rec.get('sources', [{}])[0].get('source')
    if src is None:
        raise ValueError(f"No source found for record {record.get('id')}")
    src = Path(input_dir) / src
    ext = os.path.splitext(src)[1]

    start = record.get('start', 0)
    duration = record.get('duration')
    sampling_rate = rec.get('sampling_rate')
    channels = len(rec.get('channel_ids', [0]))

    part_subset, spk, _, basename = record['id'].split('/')
    new_id = "/".join([part_subset, spk, basename])

    out_path = Path(output_dir) / f"{new_id}{ext}"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        'ffmpeg',
        '-y',
        '-hide_banner',
        '-loglevel', 'error',
        '-i', src,
        '-ss', str(start),
        '-t', str(duration),
    ]
    if sampling_rate:
        cmd += ['-ar', str(sampling_rate)]
    cmd += ['-ac', str(channels), str(out_path)]

    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jsonl-gz', required=True, help="Path to the JSONL.GZ file")
    parser.add_argument('--subset', required=True, help="Subset name")
    parser.add_argument('--input-dir', required=True, help="Directory containing the original audio files to be segmented")
    parser.add_argument('--output-dir', required=True, help="Directory where segmented audio files will be saved")
    parser.add_argument('--num-workers', type=int, default=os.cpu_count(), help="Number of parallel workers for segmentation")
    args = parser.parse_args()

    def update(*args):
        pbar.update()


    # Load cut records
    records = []
    with gzip.open(args.jsonl_gz, 'rt', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))

    print(f"Loaded {len(records)} cuts; segmenting with {args.num_workers} workers...")

    pool_args = [(rec, args.input_dir, args.output_dir) for rec in records]

    pbar = tqdm(total=len(pool_args), desc=f"Segmenting {args.subset}")

    with Pool(args.num_workers) as pool:
        for arg in pool_args:
            pool.apply_async(segment_record, args=(arg,), callback=update)
        pool.close()
        pool.join()
    pbar.close()

if __name__ == '__main__':
    main()
