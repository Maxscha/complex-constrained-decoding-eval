#!/bin/bash
set -e
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


python3 -c "import nltk; nltk.download('punkt_tab')"

types=(structured unstructured)
models=(mistral mistral-instruct phi3 olmo olmo-instruct llama31-instruct llama31-base deepseek deepseek-chat deepseek-coder deepseek-coder-instruct)

# model=llama31-instruct
# type=structured

for model in ${models[@]}; do
  for type in ${types[@]}; do
    python3 prepare_ifeval_eval.py \
      --input_path="/workspace/data/ifeval/42/input/${type}.pkl"  \
      --output_path="/workspace/data/ifeval/42/output/${model}_${type}.pkl" \
      --input_replacement='/workspace/src/ifeeval/data/input_data.jsonl' \
      --output="/workspace/src/ifeeval/data/input_response_data_${model}_${type}.jsonl"


    python3 -m evaluation_main \
      --input_data=./data/input_data.jsonl \
      --input_response_data=/workspace/src/ifeeval/data/input_response_data_${model}_${type}.jsonl \
      --output_dir=./data/${model}_${type}

  done
done
exit 0