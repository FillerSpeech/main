import pandas as pd
import re
import json
import csv
import sys

# Check if input file is provided
if len(sys.argv) < 2:
    print("Usage: python script.py <input_file>")
    sys.exit(1)

input_file = sys.argv[1]
output_csv = input_file.replace('.json', '.csv')

# # Open the JSON file and load the data
with open(input_file, 'r') as f:
    data = json.load(f)

# Write data to CSV
with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
    # Define the header
    fieldnames = ['id', 'ground_truth', 'text', 'prompt', 'acc']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # Write the header
    writer.writeheader()

    # Write each entry
    for entry in data:
        writer.writerow({
            'id': entry.get('id', [''])[0],  # Extract first item from 'id' list
            'ground_truth': entry.get('ground_truth', [''])[0],  # Extract first item from 'ground_truth' list
            'text': entry.get('text', [''])[0],  # Extract first item from 'text' list
            'prompt': entry.get('prompt', [''])[0],  # Extract first item from 'prompt' list
            'acc': entry.get('acc', '')  # Extract 'acc'
        })

print(f"CSV file has been created at {output_csv}")


# 파일 로드
data = pd.read_csv(output_csv)

# Function to clean and remove <s> and </s> tags from a string
def clean_text(text):
    return re.sub(r'<\/?s>', '', text).strip()

# Clean the 'ground_truth' and 'text' columns
data['clean_ground_truth'] = data['ground_truth'].apply(clean_text)
data['clean_text'] = data['text'].apply(clean_text)

# Function to extract <well|{duration_id}|{pitch_id}> patterns with positions
def extract_well_tokens_with_positions(text):
    matches = re.finditer(r'<(.*?)\|(\w+)\|(\w+)>', text)
    tokens = []
    for match in matches:
        start_pos = match.start()
        tokens.append(start_pos)  # Save only the position
    return tokens

# Initialize counters for position accuracy
position_correct = 0
position_total = 0

# Loop through each row in the dataset
for _, row in data.iterrows():
    # Extract token positions for cleaned ground_truth and text
    gt_positions = extract_well_tokens_with_positions(row['clean_ground_truth'])
    text_positions = extract_well_tokens_with_positions(row['clean_text'])
    
    # Calculate position accuracy
    common_positions = set(gt_positions) & set(text_positions)
    position_correct += len(common_positions)
    position_total += len(gt_positions)

# Calculate overall position accuracy
position_accuracy = position_correct / position_total if position_total > 0 else 0

print(f"Position Accuracy: {position_accuracy:.2%}")


data = pd.read_csv(output_csv)

# Function to extract <well|{duration_id}|{pitch_id}> patterns with positions
def extract_well_tokens_with_positions(text):
    matches = re.finditer(r'<(.*?)\|(\w+)\|(\w+)>', text)
    tokens = []
    for match in matches:
        start_pos = match.start()
        full_token = match.group(0)  # Entire match, e.g., <well|long|medium>
        duration_id = match.group(2)  # duration_id, e.g., long
        pitch_id = match.group(3)    # pitch_id, e.g., medium
        tokens.append((start_pos, full_token, duration_id, pitch_id))
    return tokens

# Initialize counters for duration and pitch tokens
duration_correct = 0
duration_total = 0
pitch_correct = 0
pitch_total = 0

# Loop through each row in the dataset
for _, row in data.iterrows():
    # Extract tokens and their positions for both ground_truth and text
    gt_tokens = extract_well_tokens_with_positions(row['ground_truth'])
    text_tokens = extract_well_tokens_with_positions(row['text'])
    
    # Create dictionaries for easier comparison: {position: (duration_id, pitch_id)}
    gt_tokens_dict = {pos: (duration_id, pitch_id) for pos, _, duration_id, pitch_id in gt_tokens}
    text_tokens_dict = {pos: (duration_id, pitch_id) for pos, _, duration_id, pitch_id in text_tokens}
    
    # Compare only positions where both ground_truth and text have tokens
    common_positions = set(gt_tokens_dict.keys()) & set(text_tokens_dict.keys())

    for pos in common_positions:
        gt_duration, gt_pitch = gt_tokens_dict[pos]
        text_duration, text_pitch = text_tokens_dict[pos]

        # Compare duration tokens
        duration_total += 1
        if gt_duration == text_duration:
            duration_correct += 1

        # Compare pitch tokens
        pitch_total += 1
        if gt_pitch == text_pitch:
            pitch_correct += 1

# Calculate accuracies
duration_accuracy = duration_correct / duration_total if duration_total > 0 else 0
pitch_accuracy = pitch_correct / pitch_total if pitch_total > 0 else 0

print(f"Duration Token Accuracy: {duration_accuracy:.2%}")
print(f"Pitch Token Accuracy: {pitch_accuracy:.2%}")
