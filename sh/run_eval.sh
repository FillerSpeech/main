# # # # # # # # # # # # # # # # # # # # Evaluation # # # # # # # # # # # # # # # # # # # #

# Accuracy evaluation
JSON_PATH= # inference best.json file path

python ./filler_prediction/run_eval.py ${JSON_PATH}


GPT Score evaluation
python ./filler_prediction/gpt_score/split_json.py ${JSON_PATH}

SCORE_PATH=./filler_prediction/gpt_score/json/eval_test_epoch_best.json
CHATGPT_API_KEY=YOUR_CHATGPT_API_KEY

# GPT Score-P
python ./filler_prediction/gpt_score/run_gpt_score_P.py ${SCORE_PATH} --api_key ${CHATGPT_API_KEY}

# GPT Score-T
python ./filler_prediction/gpt_score/run_gpt_score_T.py ${SCORE_PATH} --api_key ${CHATGPT_API_KEY}






