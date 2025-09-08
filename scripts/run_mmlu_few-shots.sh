#!/bin/bash set -e 

cd src/mmlu

MODELS=("phi3" "mistral" "mistral-instruct" "olmo" "olmo-instruct" "llama31-instruct" "llama31-base" "deepseek" "deepseek-chat" "deepseek-coder" "deepseek-coder-instruct")
result_dataframe='../../data/mmlu_few_shots_results_new.pkl'
K_SHOT=(1 4 5)
# K_SHOT=(6 9)


for MODEL in ${MODELS[@]}; do
    for k in ${K_SHOT[@]}; do
        echo "Evaluating ${MODEL}"
        python3 eval.py -m ${MODEL} --results_df $result_dataframe --ntrain $k --data_dir /workspace/data
        python3 eval.py -m ${MODEL} -t structured --results_df $result_dataframe --ntrain $k --data_dir /workspace/data
    done
done
