#!/bin/bash
set -e


FOLDER=${1:-"data/squad_unadapted"}
NUM_JOBS_PARALLEL=$(nvidia-smi -L | wc -l)
SEEDS=(42)
K_SHOT=(9)

JOBS=(
    "mistral unstructured 12"
    "mistral-instruct unstructured 12"
    "phi3 unstructured 12"
    "olmo unstructured 12"
    "olmo-instruct unstructured 12"
    "llama31-instruct unstructured 7" 
    "llama31-base unstructured 4"
    "deepseek unstructured 12"
    "deepseek-chat unstructured 2"
    "deepseek-coder unstructured 12"
    "deepseek-coder-instruct unstructured 2"
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
            for K in ${K_SHOT[@]}; do
                INPUT_FILE=${FOLDER}/${SEED}/input/${TYPE}_${PROMPT}_${K}.pkl
                # WHen file not exist create it
                if [ ! -f $INPUT_FILE ]; then
                    python3 src/prepare_data_squad.py \
                        --output $INPUT_FILE \
                        --template "${TYPE}_${PROMPT}" \
                        --shuffle True \
                        --random_seed $SEED \
                        --split "validation" \
                        --k_shot $K
                fi
            done
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

    INPUT_FILE=${FOLDER}/${SEED}/input/${TYPE}_${PROMPT}_${K}.pkl
    OUTPUT_FILE=${FOLDER}/${SEED}/output/${MODEL}_${TYPE}_${PROMPT}_${K}.pkl

    if [ $TYPE == "structured" ]; then
        echo "Should not happen"
        # commands+=("src/generate.py --input_files $INPUT_FILE --output_files $OUTPUT_FILE --model $MODEL --use_logits_processor True")
    else
        commands+=("src/generate.py --input_files $INPUT_FILE --output_files $OUTPUT_FILE --model $MODEL --use_logits_processor True")
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

# parallel -j $NUM_JOBS_PARALLEL --ungroup -k run_with_parallel {} {%} {#} ::: "${commands[@]}"

echo "Finished running in $(($(date +%s) - $start_time)) seconds"

python3 src/squad/eval_squad.py --data_path $FOLDER/42