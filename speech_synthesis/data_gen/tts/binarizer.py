from functools import partial

import numpy as np

import utils.commons.single_thread_env  # NOQA
from utils.audio.cwt import get_lf0_cwt, get_cont_lf0
from utils.audio.pitch.utils import f0_to_coarse
from utils.audio.pitch_extractors import extract_pitch_simple
from utils.commons.hparams import hparams
from utils.commons.indexed_datasets import IndexedDatasetBuilder
from utils.commons.multiprocess_utils import multiprocess_run_tqdm
from data_gen.tts.base_binarizer import BaseBinarizer
from pathlib import Path
import json
np.seterr(divide="ignore", invalid="ignore")

def get_json(json_file):
    with open(json_file) as f:
        return json.load(f)



class BinarizationError(Exception):
    pass

class LHBinarizer(BaseBinarizer):
    def __init__(self, processed_data_dir=None):
        if processed_data_dir is None:
            processed_data_dir = hparams["processed_data_dir"]
        self.processed_data_dir = processed_data_dir
        self.binarization_args = hparams["binarization_args"]
        self.items = {}
        self.item_names = []


    @property
    def train_item_names(self):
        with open(self.binarization_args["train_list"]) as f:
            return [l.strip() for l in f.readlines()]
    @property
    def valid_item_names(self):
        with open(self.binarization_args["valid_list"]) as f:
            return [l.strip() for l in f.readlines()]
    @property
    def test_item_names(self):
        with open(self.binarization_args["test_list"]) as f:
            return [l.strip() for l in f.readlines()]
    def process(self):
        self.load_meta_data()
        self.process_data("valid")
        self.process_data("test")
        # self.process_data("train")

    @staticmethod
    def process_pitch(wav, mel):
        f0 = extract_pitch_simple(wav)
        if sum(f0) == 0:
            raise BinarizationError("Empty f0")
        assert len(mel) == len(f0), (len(mel), len(f0))
        pitch_coarse = f0_to_coarse(f0)

        uv, cont_lf0_lpf = get_cont_lf0(f0)
        logf0s_mean_org, logf0s_std_org = np.mean(cont_lf0_lpf), np.std(
            cont_lf0_lpf
        )
        cont_lf0_lpf_norm = (cont_lf0_lpf - logf0s_mean_org) / logf0s_std_org
        cwt_spec, scales = get_lf0_cwt(cont_lf0_lpf_norm)
        return f0, cwt_spec, logf0s_mean_org, logf0s_std_org, pitch_coarse

    @classmethod
    def process_item(cls, item, binarization_args):
        item["ph_len"] = len(item["ph_token"])
        item_name = item["item_name"]
        new = Path('/workspace/sb/dataset/filler/final/real_final/f0/') / Path(item_name).with_suffix('.npy')

        wav_fn = item["wav_fn"]
        wav, mel = cls.process_audio(wav_fn, item, binarization_args)

        f0, cwt_spec, cwt_mean, cwt_std, pitch = cls.process_pitch(wav, mel)



        return {'f0': f0, 'cwt_spec': cwt_spec, 'cwt_mean': cwt_mean, 'cwt_std': cwt_std, 'pitch': pitch}


    def process_data(self, prefix):
        data_dir = hparams["binary_data_dir"]
        builders = {f"builder_{seg}": IndexedDatasetBuilder(f"{data_dir}/{prefix}_{seg}") for seg in ['f0']}
        # builder = IndexedDatasetBuilder(f"{data_dir}/{prefix}_{seg}")
        meta_data = list(self.meta_data(prefix))
        process_item = partial(
            self.process_item, binarization_args=self.binarization_args
        )
        total_sec = 0
        items = []
        args = [{"item": item} for item in meta_data]
        for item_id, item in multiprocess_run_tqdm(
            process_item, args, desc="Processing data"
        ):
            if item is not None:
                items.append(item)

        for item in items:
            builders["builder_f0"].add_item(item)
            # for seg in ['f0']:
            #     builders[f"builder_{seg}"].add_item(item[f'{seg}'])

        for seg in ['f0']:
            builders[f"builder_{seg}"].finalize()
        print(f"| {prefix} total duration: {total_sec:.3f}s")

