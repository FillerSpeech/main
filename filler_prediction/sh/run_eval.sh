# # # # # # # # # # # # # # # # # # # # Evaluation # # # # # # # # # # # # # # # # # # # #

# Accuracy evaluation
JSON_PATH=./logs/filler_pred/20241206_all_rel_f0_v5_test03/eval_test_epoch_best.json # inference best.json file path  

python ./run_eval.py ${JSON_PATH}


# GPT Score evaluation
python ./gpt_score/split_json.py ${JSON_PATH}

SCORE_PATH=./gpt_score/json/eval_test_epoch_best.json
CHATGPT_API_KEY=YOUR_CHATGPT_API_KEY

# GPT Score-P
python ./gpt_score/run_gpt_score_P.py ${SCORE_PATH} --api_key ${CHATGPT_API_KEY}

# GPT Score-T
python ./gpt_score/run_gpt_score_T.py ${SCORE_PATH} --api_key ${CHATGPT_API_KEY}






