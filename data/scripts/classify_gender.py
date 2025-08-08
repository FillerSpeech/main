
import os
import torch
from typing import List, Optional, Union, Dict
import torchaudio
from torch.utils.data import DataLoader
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    Wav2Vec2Processor
)
import glob
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.nn import functional as F

label2id = {
    "female": 0,
    "male": 1
}

id2label = {
    0: "female",
    1: "male"
}
num_labels = 2

class CustomDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: List,
        basedir: Optional[str] = None,
        sampling_rate: int = 16000,
        max_audio_len: int = 5,
        root_dir: Optional[Path] = None,
        subset: Optional[Path] = ''
    ):
        self.dataset = dataset
        self.basedir = basedir
        self.root_dir = root_dir
        self.subset = subset

        self.sampling_rate = sampling_rate
        self.max_audio_len = max_audio_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if self.basedir is None:
            filepath = self.dataset[index]
        else:
            filepath = os.path.join(self.basedir, self.dataset[index])

        speech_array, sr = torchaudio.load(filepath)

        if speech_array.shape[0] > 1:
            speech_array = torch.mean(speech_array, dim=0, keepdim=True)

        if sr != self.sampling_rate:
            transform = torchaudio.transforms.Resample(sr, self.sampling_rate)
            speech_array = transform(speech_array)
            sr = self.sampling_rate

        len_audio = speech_array.shape[1]

        if len_audio < self.max_audio_len * self.sampling_rate:
            padding = torch.zeros(1, self.max_audio_len * self.sampling_rate - len_audio)
            speech_array = torch.cat([speech_array, padding], dim=1)
        else:
            speech_array = speech_array[:, :self.max_audio_len * self.sampling_rate]

        speech_array = speech_array.squeeze().numpy()

        return {"id": Path(filepath).relative_to(self.root_dir), "input_values": speech_array, "attention_mask": None}

class CollateFunc:
    def __init__(
        self,
        processor: Wav2Vec2Processor,
        padding: Union[bool, str] = True,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: bool = True,
        sampling_rate: int = 16000,
        max_length: Optional[int] = None,
    ):
        self.sampling_rate = sampling_rate
        self.processor = processor
        self.padding = padding
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_attention_mask = return_attention_mask
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, np.ndarray]]):
        input_values = [item["input_values"] for item in batch]
        ids = [item["id"] for item in batch]
        batch = self.processor(
            input_values,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_attention_mask=self.return_attention_mask
        )

        return {
            "ids": ids,
            "input_values": batch.input_values,
            "attention_mask": batch.attention_mask if self.return_attention_mask else None
        }


def main(args):
    wavs = glob.glob(str(Path(args.input_dir) / f'audio/{args.subset}/*/*.flac'))
    wav_dir = Path(args.input_dir) / 'audio'

    print(f"# wavs: {len(wavs)}  || Batch size: {16} || {len(wavs) / 16}")

    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model)
    model = AutoModelForAudioClassification.from_pretrained(
        pretrained_model_name_or_path=args.model,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    )

    test_dataset = CustomDataset(wavs, max_audio_len=5, root_dir=wav_dir, subset=args.subset)  # for 5-second audio

    data_collator = CollateFunc(
        processor=feature_extractor,
        padding=True,
        sampling_rate=16000,
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=16,
        collate_fn=data_collator,
        shuffle=False,
        num_workers=8
    )

    model.cuda()
    model.eval()

    output_file = Path(args.input_dir) / f"gender/{args.subset}.txt"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Classifying gender"):
                input_values, attention_mask = batch['input_values'].cuda(), batch['attention_mask'].cuda()
                ids = batch["ids"]

                logits = model(input_values, attention_mask=attention_mask).logits
                scores = F.softmax(logits, dim=-1)

                pred = torch.argmax(scores, dim=1).cpu().detach().numpy()

                for id_, gender in zip(ids, pred):
                    f.write(f"{id_}\t{gender}\n")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech")
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--subset', type=Path, required=True)
    args = parser.parse_args()

    main(args)

