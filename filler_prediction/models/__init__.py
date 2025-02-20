from .filler_llm import FillerLLM

def load_model(config):
    return FillerLLM.from_config(config)