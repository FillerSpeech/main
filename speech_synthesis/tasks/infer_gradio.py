import torch
import yaml
import gradio as gr

from models.tts.fillerspeech.fillerspeech import FillerSpeech
from tasks.tts.fillerspeech import FillerSpeechTask

def load_model_and_task(checkpoint_path="./ckpts/best.pt", config_path="./ckpts/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_args = config.get("model", {})
    model = FillerSpeech(**model_args)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    task = FillerSpeechTask(model, config)
    return task


task = load_model_and_task()


def synthesize_text(text: str):
    audio, sample_rate = task.synthesize(text)
    return (sample_rate, audio)


iface = gr.Interface(
    fn=synthesize_text,
    inputs=[gr.Textbox(lines=2, placeholder="Input Text"),
            gr.Audio(source="upload", type="numpy", label="Reference Audio for Speaker")],
    outputs=gr.Audio(type="numpy"),
    title="FillerSpeech Demo",
    description=""
)

##########################
task_cls = FillerSpeechTask()

if __name__ == "__main__":
    iface.launch()