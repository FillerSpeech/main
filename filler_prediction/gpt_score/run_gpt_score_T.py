import openai
import json
import time
from tqdm import tqdm
import statistics
import argparse
import os

def load_json(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError as e:
        print(f"❌ JSON file load fail: {e}")
    except FileNotFoundError:
        print(f"❌ No file: {file_path}")
    except Exception as e:
        print(f"❌ Error: {e}")
    return []

def evaluate_filler_naturalness(sentence, client):
    SYSTEM_PROMPT_TYPE = """
    You are an expert evaluator of filler types in natural speech.
    I need your help to evaluate the performance of a model in a filler prediction scenario. 
    The model receives a target sentence and generates a response by inserting fillers of specific types at particular positions.

    **Your task is to rate the model’s response based only on the naturalness and appropriateness of the filler types used in the sentence.**  
    Consider the following aspects:

    1. **Contextual Suitability**: Assess whether the chosen filler types (e.g., "um," "oh," "yeah") fit naturally within the conversational context of the sentence, enhancing the flow and coherence.
    2. **Human-like Selection**: Determine if the filler type corresponds to what a human speaker would likely use in the given situation, considering the tone, intent, and conversational style of the sentence.

    ### **Scoring Guidelines**
    Provide a **single score** on a scale from **1 to 5**, where:

    - **1**: Poor  
      - Filler types are unnatural or disrupt the conversational flow.  
    - **2**: Below Average  
      - Some filler types seem out of place or could be improved.  
    - **3**: Neutral  
      - Filler types are acceptable but do not necessarily enhance the sentence.  
    - **4**: Good  
      - Fillers are mostly well-chosen, making the sentence sound natural.  
    - **5**: Excellent  
      - Filler types are **perfectly suited**, improving the conversational tone.  

    **Important**: Focus **only** on the filler type selection, not the placement.  
    Ignore grammar, word choice, and meaning—evaluate only whether the **type of fillers** used is what a human would naturally say.

    After evaluating, output the score **only as a number** (e.g., `4`).  
    Do NOT include explanations, extra text, or JSON formatting.
    """
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_TYPE},  
        {"role": "user", "content": f"Evaluate the following sentence:\n'{sentence}'"} 
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.5,
            max_tokens=5
        )
        response_text = response.choices[0].message.content.strip()
        try:
            return int(response_text)
        except ValueError:
            print(f"❌ Score fail: {response_text}")
    except Exception as e:
        print(f"❌ API error: {e}")
    return None

def process_filler_evaluation_text(file_path, api_key):
    # Initialize the OpenAI client using the provided API key
    client = openai.OpenAI(api_key=api_key)
    data = load_json(file_path)
    
    scores = []
    for entry in tqdm(data):
        if "text" in entry:
            score = evaluate_filler_naturalness(entry["text"], client)
            entry["ChatGPT score-T"] = score
            if score is not None:
                scores.append(score)
    
    average_score = round(statistics.mean(scores), 2) if scores else None
    print(f"\n📊 'ChatGPT score-T': {average_score}")
    
    # Construct output file name based on the input file name parts
    target_name = file_path.split('/')[-1].split('_')
    output_file = f"./gpt_score/results/{target_name[0]}_{target_name[1]}_T.json"
    
    # Create results directory if it doesn't exist
    os.makedirs("./gpt_score/results", exist_ok=True)
    
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"❌ Error: {e}")
    
    results_file = "./gpt_score/results/results.txt"
    try:
        with open(results_file, "a", encoding="utf-8") as f:
            f.write(f"{os.path.basename(output_file)}: {average_score}\n")
        print(f"✅ Saved in '{results_file}'")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate filler naturalness using ChatGPT API.")
    parser.add_argument("json_path", type=str, help="Path to the input JSON file")
    parser.add_argument("--api_key", type=str, required=True, help="ChatGPT API key")
    args = parser.parse_args()

    process_filler_evaluation_text(args.json_path, args.api_key)
