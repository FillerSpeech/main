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

import logging
import json
import contextlib
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import LlamaTokenizer, StoppingCriteriaList, AutoTokenizer 
from peft import LoraConfig, TaskType, get_peft_model 

from filler_prediction.dist_utils import get_rank

from .modeling_llama import LlamaForCausalLM
from .utils import StoppingCriteriaSub, concat_all_gather, all_gather_with_grad
from funasr import AutoModel
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
import numpy as np
import re



class FillerLLM(nn.Module):
    @property
    def device(self):
        return list(self.parameters())[0].device
    
    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def __init__(
        self,
        filler_mode="",
        pitch_type="rel_f0",
        use_extracted_feature=False,
        llama_path="",
        freeze_lora=False,

        lora=True,
        lora_rank=8,
        lora_alpha=32,
        lora_dropout=0.1,
        
        multi_prompt=False,
        dialogue_prompt=False,
        prompt_path="",
        prompt_template="",
        max_txt_len=128,
        end_sym="</s>",
        low_resource=False,  # use 8 bit
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
        infer_mode=False,
    ):
        super().__init__()
        self.filler_mode = filler_mode
        self.pitch_type = pitch_type
        self.use_extracted_feature = use_extracted_feature
        self.lora = lora
        
        self.multi_prompt = multi_prompt
        self.dialogue_prompt = dialogue_prompt
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym
        self.low_resource = low_resource
        
        self.infer_mode = infer_mode
        
        logging.info('Loading LLaMA Tokenizer')
        
        if self.infer_mode :
            from transformers import AutoTokenizer

            self.llama_tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")            
            self.llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
        else :
            if 'Llama' in llama_path: # Llama-3.1-8B-Instruct
                self.llama_tokenizer = AutoTokenizer.from_pretrained(llama_path, use_fast=False)
                self.llama_tokenizer.pad_token = "<|finetune_right_pad_id|>"
                self.llama_tokenizer.pad_token_id = 128004
                print(f'self.llama_tokenizer.pad_token : {self.llama_tokenizer.pad_token}')
                print(f'self.llama_tokenizer.pad_token_id : {self.llama_tokenizer.pad_token_id}')
                
            else :
                self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_path, use_fast=False)
                self.llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                
        self.llama_tokenizer.padding_side = "right"
        self.llama_path = llama_path
                
        
        logging.info('Loading LLaMA Model')


        if self.infer_mode:
            from transformers import AutoModelForCausalLM
            
            # HuggingFace 모델 로드
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                "lmsys/vicuna-7b-v1.5", 
                torch_dtype=torch.float16
            )
            print("Infer mode: Vicuna model loaded using AutoModelForCausalLM.")
        else :
            if self.low_resource:
                self.llama_model = LlamaForCausalLM.from_pretrained(
                    llama_path,
                    torch_dtype=torch.float16,
                    load_in_8bit=True,
                    device_map={"": device_8bit},
                )
            else:
                self.llama_model = LlamaForCausalLM.from_pretrained(
                    llama_path,
                    torch_dtype=torch.float16
                )
            

        self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))
        
        logging.info('Loading LLaMA Done')


        if self.lora:
            self.peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                inference_mode=False, 
                r=lora_rank, 
                lora_alpha=lora_alpha, 
                lora_dropout=lora_dropout,
            )
            self.llama_model = get_peft_model(self.llama_model, self.peft_config)

            if freeze_lora :
                # Freeze all parameters
                for param in self.llama_model.parameters():
                    param.requires_grad = False
                logging.info('LoRA Freezing')
            else :
                logging.info('LoRA Training')
            self.llama_model.print_trainable_parameters()
            
        # prepare prompts
        self.prompt_dict = {}
        if prompt_path:
            try:
                raw_prompts = json.load(open(prompt_path, "r"))
            except:
                print("Failed to load prompt! Try to use utf-8 encoding.")
                raw_prompts = json.load(open(prompt_path, "r", encoding='utf-8'))
            for task in raw_prompts.keys():
                if task in ['filler_prediction', 'filler_prediction_1', 'filler_prediction_2', 'filler_prediction_3', 'filler_prediction_4'] :
                    filted_prompts = [raw_prompt for raw_prompt in raw_prompts[task]]

                self.prompt_dict[task] = [prompt_template.format(p) for p in filted_prompts]
            print("Loading training prompts done!")


    def pad_tensor(self, tensor, pad_size):
        # tensor: [batch_size, sequence_length, feature_dimension] or [sequence_length, feature_dimension]
        if tensor.dim() == 3:
            padding = pad_size - tensor.size(1)
            if padding > 0:
                return F.pad(tensor, (0, 0, 0, padding), value=0)
            else:
                return tensor
        elif tensor.dim() == 2:
            padding = pad_size - tensor.size(1)
            if padding > 0:
                return F.pad(tensor, (0, padding), value=0)
            else:
                return tensor
        else:
            raise ValueError("Unsupported tensor dimension: {}".format(tensor.dim()))
        
    def pad_tensor_1(self, tensor, pad_size): 
        # tensor: [batch_size, sequence_length, feature_dimension] or [sequence_length, feature_dimension]
        if tensor.dim() == 3:
            padding = pad_size - tensor.size(1)
            if padding > 0:
                return F.pad(tensor, (0, 0, 0, padding), value=1)
            else:
                return tensor
        elif tensor.dim() == 2:
            padding = pad_size - tensor.size(1)
            if padding > 0:
                return F.pad(tensor, (0, padding), value=1)
            else:
                return tensor
        else:
            raise ValueError("Unsupported tensor dimension: {}".format(tensor.dim()))

         
    def filler_prompt_wrap(self, tgt_sen, tgt_sen_pos, tgt_filler, dur, f0, prompt, multi_prompt=False):

        wrapped_embeds = []
        wrapped_atts = []
        

        batch_embeds = []
        batch_atts = []

        for i, p in enumerate(prompt):
            if '<TGT_POS>' in p :
                tgt_sen[i] = tgt_sen_pos[i]
                tgt_sen[i] = re.sub(r"<.*?>", "<TGT_POS>", tgt_sen[i])
                
            p = p.replace("<TGT_SEN>", tgt_sen[i])
            
            revised_filler = tgt_filler[i].replace('|',', ')
            
            p = p.replace("<FILLER>", revised_filler)
            
            p_tokens = self.llama_tokenizer(
                p, return_tensors="pt", add_special_tokens=False
            ).to(self.device)
        
            p_embeds = self.llama_model.model.embed_tokens(p_tokens.input_ids) if not self.lora else self.llama_model.model.model.embed_tokens(p_tokens.input_ids)

            batch_embeds.append(p_embeds)
            batch_atts.append(p_tokens.attention_mask)

        max_len_embeds = max(tensor.size(1) for tensor in batch_embeds)
        max_len_atts = max(tensor.size(1) for tensor in batch_atts)
        
        batch_embeds_padded = [self.pad_tensor(embed, max_len_embeds) for embed in batch_embeds]
        batch_atts_padded = [self.pad_tensor(att, max_len_atts) for att in batch_atts]

        wrapped_embeds = torch.stack(batch_embeds_padded, dim=0)
        wrapped_atts = torch.stack(batch_atts_padded, dim=0)
        
        wrapped_embeds = wrapped_embeds.squeeze(1)
        wrapped_atts = wrapped_atts.squeeze(1)
    
        return wrapped_embeds, wrapped_atts


    def forward(self, samples, verbose=False):
                
        task = list(set(samples["task"]))
        if len(task) > 1 or "QA" in task:
            self.multi_prompt = True

        if self.prompt_dict:
            if self.multi_prompt:
                prompt = [random.choice(self.prompt_dict[task]) for task in samples["task"]]
            else:
                prompt = random.choice(self.prompt_dict[samples["task"][0]])
                
        if self.filler_mode :
            tgt_sen = samples["transcript"]
            tgt_filler = samples["filler"]
            dur = samples["duration"]
            tgt_sen_pos = samples["text"]

            if self.pitch_type == 'abs_f0':
                f0 = samples["abs_f0"]
            else : #rel_f0
                f0 = samples["rel_f0"]
                
            if self.prompt_dict:
                speech_embeds, speech_atts = self.filler_prompt_wrap(tgt_sen, tgt_sen_pos, tgt_filler, dur, f0, prompt, multi_prompt=self.multi_prompt)
        
        if self.filler_mode == 'all': 
            text = samples["text"]
            
            pattern = r"<(.*?)>"

            for j, t in enumerate(text):  
                matches = re.findall(pattern, t)  
                revised_t = t
                revised_dur = dur[j].split('|')
                revised_f0 = f0[j].split('|')
                
                for i, filler in enumerate(matches):
                    target_dur = revised_dur[i]  # GT duration
                    target_pit = revised_f0[i]   # GT pitch

                    revised_t = revised_t.replace(f"<{filler}>", f"<{filler}|{target_dur}|{target_pit}>", 1)

                text[j] = revised_t  
            
            text = [t + self.end_sym for t in text]
        
        else :
            text = [t + self.end_sym for t in samples["text"]]
        
        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(speech_atts.device) 
        
        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids) if not self.lora else self.llama_model.model.model.embed_tokens(to_regress_tokens.input_ids)
        
        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        
        empty_targets = (
            torch.ones(
                [speech_atts.shape[0], speech_atts.shape[1] + 1],
                dtype=torch.long
            ).to(speech_atts.device).fill_(-100) 
        )
        
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = speech_embeds.shape[0]
        bos = torch.ones(
            [batch_size, 1],
            dtype=to_regress_tokens.input_ids.dtype,
            device=to_regress_tokens.input_ids.device,
        ) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos) if not self.lora else self.llama_model.model.model.embed_tokens(bos)
        atts_bos = speech_atts[:, :1]
        
        inputs_embeds = torch.cat([bos_embeds, speech_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, speech_atts, to_regress_tokens.attention_mask], dim=1)
        
        # calulate loss
        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
            loss = outputs.loss
            
        if verbose:
            nvocab = self.llama_model.config.vocab_size
            results = outputs.logits[:, empty_targets.size(1) - 1: -1, :].contiguous().view(-1, nvocab).argmax(dim=-1)
            labels = targets[:, empty_targets.size(1):].contiguous().view(-1)
            mask = (labels != -100)
            correct = (results[mask] == labels[mask]).float().sum()
            total = len(labels[mask])

        if verbose:
            return {"loss": loss, "correct": correct, "total": total}

        return {"loss": loss}



    def generate(self, samples, generate_cfg, prompts=None):
        if self.filler_mode :
            tgt_sen = samples["transcript"]

            tgt_filler = samples["filler"]
            dur = samples["duration"]
            tgt_sen_pos = samples["text"]

            if self.pitch_type == 'abs_f0':
                f0 = samples["abs_f0"]
            else : #rel_f0
                f0 = samples["rel_f0"]

            if self.prompt_dict:
                # speech_embeds, speech_atts, lora_text_mask, lora_emotion_mask = self.filler_prompt_wrap(tgt_sen, tgt_sen_pos, tgt_filler, dur, f0, prompts, multi_prompt=self.multi_prompt)
                speech_embeds, speech_atts = self.filler_prompt_wrap(tgt_sen, tgt_sen_pos, tgt_filler, dur, f0, prompts, multi_prompt=self.multi_prompt)

        batch_size = speech_embeds.shape[0]

        bos = torch.ones(
            [batch_size, 1],
            dtype=torch.int32,
            device=speech_embeds.device,
        ) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos) if not self.lora else self.llama_model.model.model.embed_tokens(bos)
        atts_bos = speech_atts[:, :1]

        embeds = torch.cat([bos_embeds, speech_embeds], dim=1)
        attns = torch.cat([atts_bos, speech_atts], dim=1)

        stop_words_ids = [torch.tensor([2]).cuda()]
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

        if 'Llama' in self.llama_path: # Llama-3.1-8B-Instruct :
            outputs = self.llama_model.generate(
                inputs_embeds=embeds,
                max_new_tokens=generate_cfg.get("max_new_tokens", 200),
                num_beams=generate_cfg.get("num_beams", 4),
                do_sample=generate_cfg.get("do_sample", False),
                min_length=generate_cfg.get("min_length", 1),
                temperature=generate_cfg.get("temperature", 1.0),
                top_p=generate_cfg.get("top_p", 0.9),
                repetition_penalty=generate_cfg.get("repetition_penalty", 1.0),
                length_penalty=generate_cfg.get("length_penalty", 1.0),
                attention_mask=attns,
                pad_token_id=128004,
                )
        else :
            outputs = self.llama_model.generate(
                inputs_embeds=embeds,
                max_new_tokens=generate_cfg.get("max_new_tokens", 200),
                stopping_criteria=stopping_criteria,
                num_beams=generate_cfg.get("num_beams", 4),
                do_sample=generate_cfg.get("do_sample", False),
                min_length=generate_cfg.get("min_length", 1),
                temperature=generate_cfg.get("temperature", 1.0),
                top_p=generate_cfg.get("top_p", 0.9),
                repetition_penalty=generate_cfg.get("repetition_penalty", 1.0),
                length_penalty=generate_cfg.get("length_penalty", 1.0),
                attention_mask=attns,
                )

        text = self.llama_tokenizer.batch_decode(outputs, add_special_tokens=False)

        return text
    
    def infer(self, samples, generate_cfg, prompts=None):
        if self.filler_mode :
            tgt_sen = samples["input"]
            tgt_filler = samples["filler"]
            
        batch_embeds = []
        batch_atts = []
        
        for i, p in enumerate(prompts):
            if '<TGT_POS>' in p :
                tgt_sen[i] = re.sub(r"<.*?>", "<TGT_POS>", tgt_sen[i])
                
            p = p.replace("<TGT_SEN>", tgt_sen[i])
            
            if tgt_filler != "" :
                revised_filler = tgt_filler[i].replace('|',', ')
                p = p.replace("<FILLER>", revised_filler)
                
            p_tokens = self.llama_tokenizer(
                p, return_tensors="pt", add_special_tokens=False
            ).to(self.device)
        
            p_embeds = self.llama_model.model.embed_tokens(p_tokens.input_ids) if not self.lora else self.llama_model.model.model.embed_tokens(p_tokens.input_ids)

            batch_embeds.append(p_embeds)
            batch_atts.append(p_tokens.attention_mask)
            
                
        max_len_embeds = max(tensor.size(1) for tensor in batch_embeds)
        max_len_atts = max(tensor.size(1) for tensor in batch_atts)
        
        batch_embeds_padded = [self.pad_tensor(embed, max_len_embeds) for embed in batch_embeds]
        batch_atts_padded = [self.pad_tensor(att, max_len_atts) for att in batch_atts]


        speech_embeds = torch.stack(batch_embeds_padded, dim=0)
        speech_atts = torch.stack(batch_atts_padded, dim=0)
        
        speech_embeds = speech_embeds.squeeze(1)
        speech_atts = speech_atts.squeeze(1)

        batch_size = speech_embeds.shape[0]
        
        bos = torch.ones(
            [batch_size, 1],
            dtype=torch.int32,
            device=speech_embeds.device,
        ) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos) if not self.lora else self.llama_model.model.model.embed_tokens(bos)
        atts_bos = speech_atts[:, :1]

        embeds = torch.cat([bos_embeds, speech_embeds], dim=1)
        attns = torch.cat([atts_bos, speech_atts], dim=1)
    
        stop_words_ids = [torch.tensor([2]).cuda()]  
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        
        if 'Llama' in self.llama_path: # Llama-3.1-8B-Instruct :
            outputs = self.llama_model.generate(
                inputs_embeds=embeds,
                max_new_tokens=generate_cfg.get("max_new_tokens", 200),
                num_beams=generate_cfg.get("num_beams", 4),
                do_sample=generate_cfg.get("do_sample", False),
                min_length=generate_cfg.get("min_length", 1),
                temperature=generate_cfg.get("temperature", 1.0),
                top_p=generate_cfg.get("top_p", 0.9),
                repetition_penalty=generate_cfg.get("repetition_penalty", 1.0),
                length_penalty=generate_cfg.get("length_penalty", 1.0),
                attention_mask=attns,
                pad_token_id=128004,
                )
        else :
            outputs = self.llama_model.generate(
                inputs_embeds=embeds,
                max_new_tokens=generate_cfg.get("max_new_tokens", 200),
                stopping_criteria=stopping_criteria,
                num_beams=generate_cfg.get("num_beams", 4),
                do_sample=generate_cfg.get("do_sample", False),
                min_length=generate_cfg.get("min_length", 1),
                temperature=generate_cfg.get("temperature", 1.0),
                top_p=generate_cfg.get("top_p", 0.9),
                repetition_penalty=generate_cfg.get("repetition_penalty", 1.0),
                length_penalty=generate_cfg.get("length_penalty", 1.0),
                attention_mask=attns,
                )
                
        text = self.llama_tokenizer.batch_decode(outputs, add_special_tokens=False)

        return text
    
    def load_checkpoint(self, ckpt_path, filter_keys, exclude_keys=None):
        if not ckpt_path:
            logging.info("Checkpoint path not provided.")
            return {}
      
        print("Load FillerLLM checkpoint from: {}".format(ckpt_path))
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = {k: v for k, v in ckpt['model'].items() if any(key in k for key in filter_keys)}
        if exclude_keys:
            state_dict = {k: v for k, v in state_dict.items() if not any(key in k for key in exclude_keys)}
            
        return state_dict

    @classmethod
    def from_config(cls, config, infer_mode=False):
        filler_mode = config.get("filler_mode","")
        pitch_type = config.get("pitch_type", "rel_f0")
        use_extracted_feature = config.get("use_extracted_feature")
        llama_path = config.get("llama_path")
        freeze_lora = config.get("freeze_lora", False)

        lora = config.get("lora", True)
        lora_rank = config.get("lora_rank", 8)
        lora_alpha = config.get("lora_alpha", 32)
        lora_dropout = config.get("lora_dropout", 0.1)
        
        multi_prompt = config.get("multi_prompt", False)
        prompt_path = config.get("prompt_path", "")
        prompt_template = config.get("prompt_template", "")
        max_txt_len = config.get("max_txt_len", 128)
        end_sym = config.get("end_sym", "</s>")
        low_resource = config.get("low_resource", False)
        device_8bit = config.get("device_8bit", 0)

        model = cls(
            filler_mode=filler_mode,
            pitch_type=pitch_type,
            use_extracted_feature=use_extracted_feature,
            llama_path=llama_path,
            freeze_lora=freeze_lora,
            lora=lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            multi_prompt=multi_prompt,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
            infer_mode=infer_mode,
        )
                

        complete_state_dict = {}


        lora_ckpt_path = config.get("lora_ckpt", "")
        if lora_ckpt_path:
            lora_state_dict = model.load_checkpoint(lora_ckpt_path, ['llama_model'])
            complete_state_dict.update(lora_state_dict)

        if complete_state_dict:
            model.load_state_dict(complete_state_dict, strict=False)
            logging.info("Model state dict loaded successfully")
        else:
            logging.info("No state dict to load into the model")

       
        return model
