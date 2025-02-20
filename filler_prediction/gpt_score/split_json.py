import argparse
import json
import re
import os
from tqdm import tqdm

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Process JSON file for filler evaluation.")
parser.add_argument("json_path", type=str, help="Path to the input JSON file")
args = parser.parse_args()

input_path = args.json_path

# Create output directory if it doesn't exist
os.makedirs("./json", exist_ok=True)

# Use the input file's name to create the output path (e.g., ./json/eval_test_epoch_best.json)
filename = os.path.basename(input_path)
output_paths = { "filler": os.path.join("./gpt_score/json", filename) }

# Load data from the input JSON file
with open(input_path, "r", encoding="utf-8") as file:
    data = json.load(file)

# Function to clean text and replace filler tags
def process_text(text, mode):
    try:
        # Remove <s> and </s> tags
        text = re.sub(r"<s>\s*|\s*</s>", "", text)

        # Replace tags of the form <filler|duration|pitch> based on the mode
        def replace_filler(match):
            parts = match.group(1).split("|")
            if mode == "filler":
                return f"<{parts[0]}>"
        text = re.sub(r"<([^<>]+)>", replace_filler, text)
        return text
    except Exception:
        return None  

# Process data for each mode (currently only "filler")
modes = ["filler"]
processed_data = {mode: [] for mode in modes}
success_count = {mode: 0 for mode in modes} 

for entry in tqdm(data):
    try:
        text = entry.get("text", [""])[0]            # Extract the first text from the list
        ground_truth = entry.get("ground_truth", [""])[0]  # Extract the first ground_truth from the list
        entry_id = entry.get("id", [""])[0]            # Extract the first id from the list

        for mode in modes:
            processed_text = process_text(text, mode)
            processed_ground_truth = process_text(ground_truth, mode)

            if processed_text is None or processed_ground_truth is None:
                continue

            processed_entry = {
                "id": entry_id,
                "text": processed_text,
                "ground_truth": processed_ground_truth
            }

            processed_data[mode].append(processed_entry)
            success_count[mode] += 1 
    except Exception:
        pass 

# Save the processed data into a JSON file in the ./json/ folder using the input file's name
for mode in modes:
    out_path = output_paths[mode]
    with open(out_path, "w", encoding="utf-8") as file:
        json.dump(processed_data[mode], file, indent=4, ensure_ascii=False)

print("Processing complete. Files saved:")
for mode in modes:
    print(f"{output_paths[mode]}")
    print(f"Successfully processed entries for {mode}: {success_count[mode]}")
