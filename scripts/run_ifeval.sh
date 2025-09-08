#!/bin/bash
set -e

FOLDER=${1:-"data/ifeval"}

NUM_JOBS_PARALLEL=$(nvidia-smi -L | wc -l)
SEEDS=(42)
K_SHOT=(9)

JOBS=(
    "mistral structured 0"
    "mistral unstructured 0"
    "mistral-instruct structured 0"
    "mistral-instruct unstructured 0"
    "phi3 structured 0"
    "phi3 unstructured 0"
    "olmo structured 0"
    "olmo unstructured 0"
    "olmo-instruct structured 0"
    "olmo-instruct unstructured 0"
    "llama31-instruct structured 0" 
    "llama31-instruct unstructured 0" 
    "llama31-base structured 0"
    "llama31-base unstructured 0"
    "deepseek structured 0"
    "deepseek unstructured 0"
    "deepseek-chat structured 0"
    "deepseek-chat unstructured 0"
    "deepseek-coder structured 0"
    "deepseek-coder unstructured 0"
    "deepseek-coder-instruct structured 0"
    "deepseek-coder-instruct unstructured 0"
)

MODELS=()
TYPES=()
PROMPTS=()

for triplet in "${JOBS[@]}"; do
    IFS=' ' read -r -a test <<< "$triplet"
    echo "Model: ${test[0]}, Type: ${test[1]}, Prompt: ${test[2]}"
    MODELS+=(${test[0]})
    TYPES+=(${test[1]})
    PROMPTS+=(${test[2]})
done


# # # --max_examples 1000 \
echo "Creating Input Files"
for SEED in ${SEEDS[@]}; do
    for PROMPT in ${PROMPTS[@]}; do
        for TYPE in ${TYPES[@]}; do
            INPUT_FILE=${FOLDER}/${SEED}/input/${TYPE}.pkl
            # WHen file not exist create it
            if [ ! -f $INPUT_FILE ]; then
                python3 src/prepare_data_ifeval.py \
                    --output $INPUT_FILE \
                    --type $TYPE \
                    --random_seed $SEED \
                    --split "train" \
                    
            fi
        done
    done
done


commands=()
for triplet in "${JOBS[@]}"; do
    IFS=' ' read -r -a test <<< "$triplet"
    MODEL=${test[0]}
    TYPE=${test[1]}
    PROMPT=${test[2]}
    INPUT_FILES_STRING=""
    OUTPUT_FILES_STRING=""
    K=${K_SHOT[0]}
    SEED=${SEEDS[0]}

    INPUT_FILE=${FOLDER}/${SEED}/input/${TYPE}.pkl
    OUTPUT_FILE=${FOLDER}/${SEED}/output/${MODEL}_${TYPE}.pkl

    if [ $TYPE == "structured" ]; then
        commands+=("src/generate.py --input_files $INPUT_FILE --output_files $OUTPUT_FILE --model $MODEL --use_logits_processor True --max_tokens 4096")
    else
        commands+=("src/generate.py --input_files $INPUT_FILE --output_files $OUTPUT_FILE --model $MODEL --use_logits_processor False --max_tokens 4096")
    fi
done

rm -rf logs
mkdir logs


run_with_parallel() {
    local cmd="$1"
    local gpu="$2"
    local num="$3"
    NUM_JOBS_PER_GPU=1
    # echo "$cmd"

    # gpu - 1
    gpu=$(((gpu - 1) / NUM_JOBS_PER_GPU))

    echo "Running on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu python3 $cmd &>logs/$num.txt
    sleep 2
}

export -f run_with_parallel

echo "Running ${#commands[@]} commands with $NUM_JOBS_PARALLEL parallel jobs"

start_time=$(date +%s)

parallel -j $NUM_JOBS_PARALLEL --ungroup -k run_with_parallel {} {%} {#} ::: "${commands[@]}"

# echo "Finished running in $(($(date +%s) - $start_time)) seconds"
# 
# python3 src/eval_squad.py.py --data_path $FOLDER/42