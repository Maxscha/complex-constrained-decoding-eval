#!/bin/bash

set -e 

cd src/truthqa

MODELS=("phi3" "mistral" "mistral-instruct" "olmo" "olmo-instruct" "llama31-instruct" "llama31-base" "deepseek" "deepseek-chat" "deepseek-coder" "deepseek-coder-instruct")

result_dataframe='../../data/truth_results_test.pkl'

for MODEL in ${MODELS[@]}; do
    echo "Evaluating ${MODEL}"
    python3 eval.py -m ${MODEL} --results_df $result_dataframe
    python3 eval.py -m ${MODEL} -t structured --results_df $result_dataframe
done