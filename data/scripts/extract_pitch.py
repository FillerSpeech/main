import parselmouth
import numpy as np
import os
from tqdm import tqdm
import argparse
import glob
from pathlib import Path
from functools import partial
from multiprocessing import Pool
import os
import torchaudio
import glob

def get_pitch(audio, hop_size=256, audio_sample_rate=16000, f0_min=80, f0_max=800,
                  voicing_threshold=0.6):
    time_step = hop_size / audio_sample_rate * 1000
    n_mel_frames = int(len(audio) // hop_size)
    f0_pm = parselmouth.Sound(audio, audio_sample_rate).to_pitch_ac(
        time_step=time_step / 1000, voicing_threshold=voicing_threshold,
        pitch_floor=f0_min, pitch_ceiling=f0_max).selected_array['frequency']
    pad_size = (n_mel_frames - len(f0_pm) + 1) // 2
    f0 = np.pad(f0_pm, [[pad_size, n_mel_frames - len(f0_pm) - pad_size]], mode='constant')
    return f0

def process(wav, output_dir):
    audio, sr = torchaudio.load(wav)
    audio = audio.squeeze().numpy()
    assert sr == 16000, f"Different SR: {sr}"
    f0 = get_pitch(audio)
    subset, spk, basename = wav.split('/')[-3:]
    npy_file = Path(output_dir) / f"pitch/{subset}/{spk}/{Path(basename).stem}.npy"
    npy_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_file, f0)

def main(args):
    wavs = glob.glob(str(Path(args.input_dir) / 'audio/*/*/*.flac'))

    chunksize = max(1, len(wavs) // ((args.num_workers or 1) * 4))

    work = partial(process, output_dir=str(args.input_dir))
    with Pool(args.num_workers) as pool:
        for _ in tqdm(pool.imap_unordered(work, wavs, chunksize=chunksize), total=len(wavs), desc="Extracting pitch"):
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--num-workers', type=int, default=os.cpu_count())
    args = parser.parse_args()
    main(args)