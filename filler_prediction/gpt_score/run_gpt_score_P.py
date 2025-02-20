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

def evaluate_sentence(sentence, client):
    SYSTEM_PROMPT_POSITION = """
    You are an expert evaluator of filler placement. 
    I need your help to evaluate the performance of a model in a filler prediction scenario. 
    The model receives a target sentence and generates a response by inserting fillers at specific positions.
    
    **Your task is to rate the model’s response based only on the correctness of filler positions.** 
    Ignore the content of the fillers themselves and focus strictly on whether the **placement of the fillers** aligns with natural speaking patterns.

    ### **Scoring Guidelines (Evaluate only the filler position!)**
    Provide a **single score** on a scale from **1 to 5**, where:

    - **1**: Poor  
      - Fillers are placed incorrectly, disrupting the sentence’s natural flow.  
    - **2**: Below Average  
      - Some fillers are misplaced, causing minor disruptions.  
    - **3**: Neutral  
      - Fillers are placed in acceptable locations but do not necessarily enhance the sentence.  
    - **4**: Good  
      - Fillers are mostly well-placed, making the sentence sound natural.  
    - **5**: Excellent  
      - Fillers are placed **perfectly**, improving the conversational tone.  

    **Important**: Focus **only** on **filler position** for this evaluation.

    After evaluating, output the score **only as a number** (e.g., `4`).
    """
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_POSITION}, 
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
            score = int(response_text)
            return score
        except ValueError:
            print(f"❌ Score fail: {response_text}")
    except Exception as e:
        print(f"❌ API error: {e}")
    return None

def process_filler_evaluation(file_path, api_key):
    # Initialize the OpenAI client with the provided API key
    client = openai.OpenAI(api_key=api_key)
    data = load_json(file_path)
    
    scores = []
    for entry in tqdm(data):
        if "ground_truth" in entry:
            score = evaluate_sentence(entry["text"], client)
            entry["ChatGPT score-P"] = score
            if score is not None:
                scores.append(score)
    
    average_score = round(statistics.mean(scores), 2) if scores else None
    print(f"\n📊 'ChatGPT score-P': {average_score}")
    
    # Create an output file name based on the input file's name parts
    target_name = file_path.split('/')[-1].split('_')
    output_file = f"./gpt_score/results/{target_name[0]}_{target_name[1]}_P.json"
    
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
    parser = argparse.ArgumentParser(description="Filler evaluation using ChatGPT API.")
    parser.add_argument("json_path", type=str, help="Path to the input JSON file")
    parser.add_argument("--api_key", type=str, required=True, help="ChatGPT API key")
    args = parser.parse_args()

    process_filler_evaluation(args.json_path, args.api_key)
