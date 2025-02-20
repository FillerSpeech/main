# Copyright (2024) Tsinghua University, Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

from torch.utils.data import Dataset
import soundfile as sf
import numpy as np
import torch.nn.functional as F

        
class FillerDataset(Dataset):
    def __init__(self, ann_path, cfg):
        super().__init__()
        self.model_config = cfg.config.model
        self.data_config = cfg.config.datasets
        
        self.annotation = json.load(open(ann_path, "r"))["annotation"]
        self.mode =self.model_config.filler_mode

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        
        transcript = [s["transcript"] for s in samples]           
        filler = [s["filler"] for s in samples]
        text = [s["text"] for s in samples]
        task = [s["task"] for s in samples]
        id = [s["id"] for s in samples]
        duration = [s["duration"] for s in samples]
        abs_f0 = [s["abs_f0"] for s in samples]
        rel_f0 = [s["rel_f0"] for s in samples]
        

        return {
            "transcript": transcript,
            "filler": filler,
            "text": text,
            "task": task,
            "id": id,
            "duration": duration,
            "abs_f0": abs_f0,
            "rel_f0": rel_f0,
        }
   
    def __getitem__(self, index):
        ann = self.annotation[index]

        transcript = None
        filler = None
        text = None
        task = ann["task"]
         
        transcript = ann["transcript"]
        text = ann["text"]
        filler = ann["filler"]
        duration = ann["duration"]
        abs_f0 = ann["abs_f0"]
        rel_f0 = ann["rel_f0"]
                    
        return {
            "transcript": transcript,  #str
            "filler": filler,
            "text": text,
            "task": task,
            "id": ann["path"],
            "duration": duration,
            "abs_f0": abs_f0,
            "rel_f0": rel_f0,
        }