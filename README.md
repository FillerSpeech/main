# FillerSpeech: Towards Human-Like Text-to-Speech Synthesis Filler Injection and Filler Style Control [[Demo]](https://fillerspeech.github.io/main/)
### Abstract
Recent advancements in speech synthesis have significantly improved the audio quality and pronunciation of synthesized speech. To further advance toward human-like conversational speech synthesis, this paper presents FillerSpeech, a novel speech synthesis framework that enables natural filler insertion and control over filler style. To address this, we construct a filler-inclusive speech data, derived from the open-source large-scale speech corpus. This data includes fillers with pitch and duration information. For the generation and style control of natural fillers, we propose a method that tokenizes filler style and utilizes cross-attention with the input text. Furthermore, we introduce a large language model-based filler prediction method that enables natural insertion of fillers even when only text input is provided. The experimental results demonstrate that the constructed dataset is valid and that our proposed methods for filler style control and filler prediction are effective.


---
## Installation

To install the required packages, run the following command:
```bash
pip install -r requirements.txt
```

## Pre-trained model
~~~
# Filler Prediction
gdown --folder https://drive.google.com/drive/folders/1OqGAL0iVn9xDPA8kAgOD64A8xjnQzyxs

# Speech Synthesis
gdown --folder https://drive.google.com/drive/folders/1JfHPq2Li3Sjm4_T0PeavmjLE7_nXmwxB
~~~
---

# Filler Prediction
## Prompt Types
There are four types of prompts:

### Prompt 1: Specify the Desired Filler Type Only
- **Description:** The model predicts the positions, durations, and pitches of the fillers.
- **Prompt Example:**
  ```
  Target sentence: <TGT_SEN>
  USER: Add the specified fillers (like <FILLER>) to the target sentence to make it sound more natural. For each filler, also specify its duration (short, medium, long) and pitch (low, medium, high) that sound contextually appropriate and natural.
  ```
- **Task ID:** `filler_prediction_1`

### Prompt 2: Specify the Desired Filler Type and Position
- **Description:** The model predicts the duration and pitch for the fillers.
- **Prompt Example:**
  ```
  Target sentence: <TGT_SEN>
  USER: Add the specified fillers (like <FILLER>) at target positions <TGT_POS> in the target sentence. For each filler, also specify its duration (short, medium, long) and pitch (low, medium, high) that sound contextually appropriate and natural.
  ```
- **Task ID:** `filler_prediction_2`

### Prompt 3: Provide Filler Candidates Only
- **Description:** The model predicts the filler type, position, duration, and pitch.
- **Prompt Example:**
  ```
  Target sentence: <TGT_SEN>
  Filler word options: oh, ah, ha, eh, aha, huh, hm, uh, yeah, mm, um, ya, um, well
  USER: Add contextually appropriate fillers to the target sentence. For each filler, also specify its duration (short, medium, long) and pitch (low, medium, high) that sound contextually appropriate and natural.
  ```
- **Task ID:** `filler_prediction_3`

### Prompt 4: Provide Filler Candidates and Specify Positions
- **Description:** The model predicts the filler type, duration, and pitch.
- **Prompt Example:**
  ```
  Target sentence: <TGT_SEN>
  Filler word options: oh, ah, ha, eh, aha, huh, hm, uh, yeah, mm, um, ya, um, well
  USER: Add contextually appropriate fillers at target positions <TGT_POS> in the target sentence. For each filler, also specify its duration (short, medium, long) and pitch (low, medium, high) that sound contextually appropriate and natural.
  ```
- **Task ID:** `filler_prediction_4`

---

## Dataset

### Full Dataset Paths
- **Training Dataset:** `./filler_prediction/data/filler_pred_train.json`
- **Validation Dataset:** `./filler_prediction/data/filler_pred_valid.json`
- **Test Dataset:** `./filler_prediction/data/filler_pred_test.json`

### Test Dataset Paths for Each Prompt
- For Prompt 1: `./filler_prediction/data/filler_pred_test_1.json`
- For Prompt 2: `./filler_prediction/data/filler_pred_test_2.json`
- For Prompt 3: `./filler_prediction/data/filler_pred_test_3.json`
- For Prompt 4: `./filler_prediction/data/filler_pred_test_4.json`

---

## Model Training

You can modify the training settings in the `./filler_prediction/configs/config_filler_pred.yaml` file:

To start training, run:
```bash
bash ./sh/run_train.sh
```

---

## Model Inference

You can modify the inference settings in the `./filler_prediction/configs/test_config_filler_pred.yaml` file:


To start inference, run:
```bash
bash ./sh/run_test.sh
```
---

## Accuracy & GPT Score

To evaluate the model, run:
```bash
bash ./sh/run_eval.sh
```

---




