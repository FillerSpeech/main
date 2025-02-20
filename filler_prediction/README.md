# FillerSpeech

FillerSpeech is an LLM-based model that predicts the positions, types, durations, and pitches of fillers in sentences.

---

## Installation

To install the required packages, run the following command:
```bash
pip install -r requirements.txt
```

---

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
- **Training Dataset:** `./data/filler_pred_train.json`
- **Validation Dataset:** `./data/filler_pred_valid.json`
- **Test Dataset:** `./data/filler_pred_test.json`

### Test Dataset Paths for Each Prompt
- For Prompt 1: `./data/filler_pred_test_1.json`
- For Prompt 2: `./data/filler_pred_test_2.json`
- For Prompt 3: `./data/filler_pred_test_3.json`
- For Prompt 4: `./data/filler_pred_test_4.json`

---

## Model Training

You can modify the training settings in the `config_filler_pred.yaml` file:
- **LLM Path Setting:**
  ```yaml
  config.model.llama_path: <path to the downloaded Vicuna-7b-v1.5 checkpoint>
  ```
  Download Vicuna-7b-v1.5 from: [Hugging Face](https://huggingface.co/lmsys/vicuna-7b-v1.5)
- **Evaluation Mode Setting:**
  ```yaml
  config.run.evaluate: False
  ```

To start training, run:
```bash
bash ./sh/run_train.sh
```

---

## Model Inference

You can modify the inference settings in the `test_config_filler_pred.yaml` file:
- **LLM Path Setting:**
  ```yaml
  config.model.llama_path: <path to the Vicuna-7b-v1.5 checkpoint>
  ```
- **Evaluation Mode Setting:**
  ```yaml
  config.run.evaluate: False
  ```

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