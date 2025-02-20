import torch.optim
import torch.utils.data
import numpy as np
import torch
import torch.optim
import torch.utils.data
import torch.distributions
from utils.commons.dataset_utils import (
    BaseDataset,
    collate_1d_or_2d,
)
from utils.text.text_encoder import build_token_encoder
from utils.text import intersperse, intersperse_exp
import json
from data_gen.tts.txt_processors.base_text_processor import get_txt_processor_cls
from utils.audio import librosa_wav2spec



import torch.optim
import torch.utils.data
import numpy as np
import torch
import torch.optim
import torch.utils.data
import torch.distributions
from utils.audio.pitch.utils import norm_interp_f0, denorm_f0
from utils.commons.dataset_utils import (
    collate_1d_or_2d,
    collate_1d,
    collate_2d,
)
from tasks.tts.dataset_utils import BaseSpeechDataset
from utils.commons.indexed_datasets import IndexedDataset
from utils.text.text_encoder import build_token_encoder
from utils.text import intersperse, intersperse_exp


f0_lab_to_idx = {}
f0_lab_to_idx['female'] = 1
f0_lab_to_idx['male'] = 2
idx = 3
for gender in ['female', 'male']:
    for key in ['low', 'medium', 'high']:
        f0_lab_to_idx[f"{gender}_{key}"] = idx
        idx += 1

dur_lab_to_idx = {
    'none': 1,
    'short': 2,
    'medium': 3,
    'long': 4
}

def processed_audio(wav_fn, hparams):
    wav2spec_dict = librosa_wav2spec(
        wav_fn,
        fft_size=hparams["fft_size"],
        hop_size=hparams["hop_size"],
        win_length=hparams["win_size"],
        num_mels=hparams["audio_num_mel_bins"],
        fmin=hparams["fmin"],
        fmax=hparams["fmax"],
        sample_rate=hparams["audio_sample_rate"],
        loud_norm=hparams["loud_norm"],
    )
    mel = wav2spec_dict["mel"]

    return mel

class BaseForMASDataset(BaseSpeechDataset):
    def _get_item(self, index):
        if hasattr(self, "avail_idxs") and self.avail_idxs is not None:
            index = self.avail_idxs[index]
        if self.indexed_ds is None:
            self.indexed_ds = IndexedDataset(f"{self.data_dir}/{self.prefix}")
        if self.indexed_ds_cond is None:
            self.indexed_ds_cond = IndexedDataset(f"{self.data_dir}/{self.prefix}_{self.cond_type}")
        if self.indexed_ds_dur is None:
            self.indexed_ds_dur = IndexedDataset(f"{self.data_dir}/{self.prefix}_dur_tok")
        if self.indexed_ds_f0 is None:
            self.indexed_ds_f0 = IndexedDataset(f"{self.data_dir}/{self.prefix}_f0")
        if self.indexed_ds_wf is None:
            self.indexed_ds_wf = IndexedDataset(f"{self.data_dir}/{self.prefix}_{self.cond_type.replace('f0', 'wf')}")

        item = self.indexed_ds[index]
        item['f_cond'] = self.indexed_ds_cond[index]
        item_f0 = self.indexed_ds_f0[index]
        item['dur_cond'] = self.indexed_ds_dur[index]
        item['wf'] = self.indexed_ds_wf[index]

        for k, v in item_f0.items():
            item[k] = v
        return item

    def __getitem__(self, index):
        sample = super(BaseForMASDataset, self).__getitem__(index)
        item = self._get_item(index)
        hparams = self.hparams
        mel = sample["mel"]
        T = mel.shape[0]
        ph_token = sample["txt_token"]
        if hparams["use_pitch_embed"]:
            assert "f0" in item
            pitch = torch.LongTensor(item.get(hparams.get('pitch_key', 'pitch')))[:T]
            f0, uv = norm_interp_f0(item["f0"][:T])
            uv = torch.FloatTensor(uv)
            f0 = torch.FloatTensor(f0)
        else:
            f0, uv, pitch = None, None, None
        sample["f0"], sample["uv"], sample["pitch"] = f0, uv, pitch
        sample["f_cond"] = item["f_cond"]
        sample["wf"] = item["wf"]
        sample["dur_cond"] = item["dur_cond"]
        sample["mel2ph"] = torch.ones(T)
        sample["cwt_spec"] = torch.tensor(item["cwt_spec"])
        sample["f0_mean"] = item["cwt_mean"]
        sample["f0_std"] = item["cwt_std"]
        return sample

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        batch = super(BaseForMASDataset, self).collater(samples)
        hparams = self.hparams
        mel2ph = collate_1d_or_2d([s["mel2ph"] for s in samples], 0.0)
        if hparams["use_pitch_embed"]:
            f0 = collate_1d_or_2d([s["f0"] for s in samples], 0.0)
            pitch = collate_1d_or_2d([s["pitch"] for s in samples])
            uv = collate_1d_or_2d([s["uv"] for s in samples])
        else:
            f0, uv, pitch = None, None, None

        cwt_spec = collate_1d_or_2d([s["cwt_spec"] for s in samples])
        f0_mean = torch.Tensor([s["f0_mean"] for s in samples])
        f0_std = torch.Tensor([s["f0_std"] for s in samples])
        batch.update(
            {
                "pitch": pitch,
                "f0": f0,
                "uv": uv,
                "mel2ph": mel2ph,
                "cwt_spec": cwt_spec,
                "f0_mean": f0_mean,
                "f0_std": f0_std,
            }
        )
        return batch

class MatchaTTSDataset(BaseForMASDataset):
    def __init__(self, prefix, shuffle=False, items=None, data_dir=None):
        super().__init__(prefix, shuffle, items, data_dir)
        data_dir = self.hparams["processed_data_dir"]
        self.f_cond_num_for_emb = self.hparams['f_cond_num'] - 1
        self.dur_num_for_emb = 5
        self.token_encoder = build_token_encoder(f"{data_dir}/phone_set.json")

        self.cond_type = self.hparams['cond_type']
        self.indexed_ds_cond = None
        self.indexed_ds_dur = None
        self.indexed_ds_f0 = None
        self.indexed_ds_wf = None

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        ph_token = sample["txt_token"]
        ph_token = intersperse(ph_token, len(self.token_encoder))
        ph_token = torch.IntTensor(ph_token)
        sample["txt_token"] = ph_token

        f_cond = sample['f_cond']
        f_cond = intersperse_exp(f_cond)
        sample["f_cond"] = f_cond

        dur_cond = sample['dur_cond']
        dur_cond = intersperse_exp(dur_cond)
        sample["dur_cond"] = dur_cond

        sample["wf"] = torch.IntTensor(sample["wf"])

        return sample

    def collater(self, samples):
        batch = super().collater(samples)

        f_conds = collate_1d_or_2d([s["f_cond"] for s in samples], 0)
        dur_conds = collate_1d_or_2d([s["dur_cond"] for s in samples], 0)
        wfs = collate_1d_or_2d([s["wf"] for s in samples], 0)
        wf_lengths = torch.LongTensor([s["wf"].shape[-1] for s in samples])
        batch.update(
            {
                "f_conds": f_conds,
                "dur_conds": dur_conds,
                "wfs": wfs,
                "wf_lengths": wf_lengths

            }
        )
        return batch

class FillerSpeechInferenceDataset(BaseDataset):
    def __init__(self, prefix, shuffle=False, items=None, data_dir=None):
        super().__init__(shuffle)
        from utils.commons.hparams import hparams
        self.hparams = hparams
        with open(self.hparams["test_inference_list"]) as f:
            self.jsons = json.load(f)


        self.f_cond_num_for_emb = self.hparams['f_cond_num'] - 1
        self.raw_data_dir = hparams["raw_data_dir"]
        self.processed_dir = hparams["processed_data_dir"]
        self.ph_encoder = self._phone_encoder()
        self.word_encoder = self._word_encoder()

        self.hparams = hparams


        self.preprocess_args = hparams["preprocess_args"]
        txt_processor = self.preprocess_args["txt_processor"]
        self.txt_processor = get_txt_processor_cls(txt_processor)

    def txt_to_ph(self, txt_raw):
        txt_struct, txt = self.txt_processor.process(txt_raw, self.preprocess_args)
        ph = [p for w in txt_struct for p in w[1]]
        ph_gb_word = ["_".join(w[1]) for w in txt_struct]
        words = [w[0] for w in txt_struct]
        # word_id=0 is reserved for padding
        ph2word = [
            w_id + 1 for w_id, w in enumerate(txt_struct) for _ in range(len(w[1]))
        ]
        return " ".join(ph), txt, " ".join(words), ph2word, " ".join(ph_gb_word)

    def __len__(self):
        return len(self.jsons)

    def _phone_encoder(self):
        ph_set_fn = f"{self.processed_dir}/phone_set.json"
        ph_set = json.load(open(ph_set_fn, "r"))
        print("| Load phone set: ", ph_set)
        return build_token_encoder(ph_set_fn)

    def _word_encoder(self):
        word_set_fn = f"{self.processed_dir}/word_set.json"
        word_set = json.load(open(word_set_fn, "r"))
        print("| Load word set. Size: ", len(word_set), word_set[:10])
        return build_token_encoder(word_set_fn)

    def __getitem__(self, index):
        hparams = self.hparams
        item = self._getitem(index)

        spec = torch.Tensor(item["mel"])

        ph_token = intersperse(item["ph_token"], len(self.ph_encoder))
        ph_token = torch.tensor(ph_token, dtype=torch.int32)

        f_cond = intersperse_exp(item['f_cond'].numpy())

        dur_cond = intersperse_exp(item['duration'].numpy())

        wf = torch.tensor(item["wf"], dtype=torch.int32)
        dur_w_tok = torch.tensor(item["dur_w_tok"], dtype=torch.int32)


        sample = {
            "id": index,
            "item_name": item["item_name"],
            "text": item["txt"],
            "txt_token": ph_token,
            "mel": spec,
            "mel_nonpadding": spec.abs().sum(-1) > 0,
            "f_cond": f_cond,
            "dur_cond": dur_cond,
            "wf": wf,
            "w_dur": dur_w_tok
        }

        sample["f0"], sample["uv"], sample["pitch"] = None, None, None

        return sample

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        hparams = self.hparams
        id = torch.LongTensor([s["id"] for s in samples])
        item_names = [s["item_name"] for s in samples]
        text = [s["text"] for s in samples]
        txt_tokens = collate_1d_or_2d([s["txt_token"] for s in samples], 0)
        mels = collate_1d_or_2d([s["mel"] for s in samples], 0.0)
        txt_lengths = torch.LongTensor([s["txt_token"].numel() for s in samples])
        mel_lengths = torch.LongTensor([s["mel"].shape[0] for s in samples])


        batch = {
            "id": id,
            "item_name": item_names,
            "nsamples": len(samples),
            "text": text,
            "txt_tokens": txt_tokens,
            "txt_lengths": txt_lengths,
            "mels": mels,
            "mel_lengths": mel_lengths,
        }

        f_conds = collate_1d_or_2d([s["f_cond"] for s in samples], 0)
        wfs = collate_1d_or_2d([s["wf"] for s in samples], 0)
        wf_lengths = torch.LongTensor([s["wf"].shape[-1] for s in samples])
        dur_conds = collate_1d_or_2d([s["dur_cond"] for s in samples], 0)
        w_durs = collate_1d_or_2d([s["w_dur"] for s in samples], 0)
        w_dur_lengths = torch.LongTensor([s["w_dur"].shape[-1] for s in samples])



        batch.update(
            {
                "f_conds": f_conds,
                "wfs": wfs,
                "wf_lengths": wf_lengths,
                "dur_conds": dur_conds,
                "w_durs": w_durs,
                "w_durs_lengths": w_dur_lengths
            }
        )

        return batch

    def _getitem(self, index):
        j = self.jsons[index]
        txt_raw = j['text']
        wav = j['wav']
        filename = j['filename']
        gender = j['gender']
        fillers = [fil for fil in j['fillers'] if fil['type'] == 'N']

        cond_type = self.hparams["cond_type"]

        ph, txt, word, ph2word, ph_gb_word = self.txt_to_ph(txt_raw)

        word_token = self.word_encoder.encode(word)
        ph_token = self.ph_encoder.encode(ph)
        mel = processed_audio(wav, self.hparams)

        words = word.split()
        ph2word = torch.LongTensor(ph2word)

        length = len(ph_token)

        dur_tokens = torch.zeros(length, dtype=torch.int32)

        dur_tokens[:] = dur_lab_to_idx['none']

        f_cond_tokens = torch.zeros(length, dtype=torch.int32)
        f_cond_tokens[:] = f0_lab_to_idx[f"{gender}"]

        start_index = 0
        for i in fillers:
            if i['type'] == 'N':
                index = words.index(f"<{i['filler']}>", start_index)
                dur_tokens[ph2word == index + 1] = dur_lab_to_idx[f"{i['duration']}"]
                f_cond_tokens[ph2word == index + 1] = f0_lab_to_idx[f"{gender}_{i[cond_type]}"]
                start_index = index + 1


        json_text = j["text"]

        wf_tokens = torch.zeros(len(json_text.split()), dtype=torch.int32)
        dur_w_tokens = torch.zeros(len(json_text.split()), dtype=torch.int32)

        wf_tokens[:] = f0_lab_to_idx[f"{gender}"]
        dur_w_tokens[:] = dur_lab_to_idx['none']

        filler_idx = 0
        for idx, tw in enumerate(json_text.split()):
            if '<mm>' in tw or '<Mm>' in tw:
                continue
            elif "<" in tw:
                if filler_idx >= len(fillers):
                    print(j)
                    print(fillers)
                    print()
                wf_tokens[idx] = f0_lab_to_idx[f"{gender}_{fillers[filler_idx][cond_type]}"]
                dur_w_tokens[idx] = dur_lab_to_idx[f"{fillers[filler_idx]['duration']}"]
                filler_idx += 1


        return {
            "txt": txt,
            "txt_raw": txt_raw,
            "ph": ph,
            "word": word,
            "ph2word": ph2word,
            "ph_gb_word": ph_gb_word,
            "wav_fn": str(wav),
            "wav_align_fn": str(wav),
            "word_token": word_token,
            "ph_token": ph_token,
            "mel": mel,
            "item_name": filename,
            "f_cond": f_cond_tokens,
            "duration":  dur_tokens,
            "wf": wf_tokens.numpy(),
            'dur_w_tok': dur_w_tokens.numpy()
        }

