#!/bin/bash
set -e

FOLDER=${1:-"data/squad_train_multi_new_new_new"}
MAX_TRAIN_SAMPLES=1000

NUM_JOBS_PARALLEL=$(nvidia-smi -L | wc -l)

# Run prompt experiments
# MODELS=("llama2" "llama2-chat" "code-llama" "code-llama-instruct" "llama3-base" "llama3" "phi3" "mistral" "mistral-instruct" "olmo" "olmo-instruct")


MODELS=("phi3" "mistral" "mistral-instruct" "olmo" "olmo-instruct" "llama31-instruct" "llama31-base" "deepseek" "deepseek-chat" "deepseek-coder" "deepseek-coder-instruct")
TYPES=(structured)
# Maybe change the naming as well
PROMPTS=(1 2 3 4 5 6 7 8 9 10 11 12 13)
SEEDS=(42)
K_SHOT=(9)

# Make this smarter split up creation

INPUT_FILES_STRUCTURED=()
INPUT_FILES_UNSTRUCTURED=()

echo "Creating Input Files"
for SEED in ${SEEDS[@]}; do
    for PROMPT in ${PROMPTS[@]}; do
        for TYPE in ${TYPES[@]}; do
            for K in ${K_SHOT[@]}; do
                INPUT_FILE=${FOLDER}/${SEED}/input/${TYPE}_${PROMPT}.pkl
                # WHen file not exist create it
                if [ ! -f $INPUT_FILE ]; then
                    python3 src/prepare_data_squad_multi.py \
                        --output $INPUT_FILE \
                        --template "${TYPE}_${PROMPT}" \
                        --shuffle True \
                        --random_seed $SEED \
                        --max_examples $MAX_TRAIN_SAMPLES \
                        --split "train"
                fi
                if [ $TYPE == "structured" ]; then
                    INPUT_FILES_STRUCTURED+=($INPUT_FILE)
                else
                    INPUT_FILES_UNSTRUCTURED+=($INPUT_FILE)
                fi
            done
        done
    done
done


echo "Created ${#INPUT_FILES_STRUCTURED[@]} Structured Files"
echo "Created ${#INPUT_FILES_UNSTRUCTURED[@]} Unstructured Files"

commands=()
for MODEL in ${MODELS[@]}; do
    # Create command for structured files
    INPUT_FILES_STRING=""
    OUTPUT_FILES_STRING=""
    for INPUT_FILE in ${INPUT_FILES_STRUCTURED[@]}; do
        OUTPUT_FILE=${INPUT_FILE//input\//output\/${MODEL}_}
        INPUT_FILES_STRING+="$INPUT_FILE "
        OUTPUT_FILES_STRING+="$OUTPUT_FILE "
    done
    commands+=("src/generate.py --input_files $INPUT_FILES_STRING --output_files $OUTPUT_FILES_STRING --model $MODEL --use_logits_processor True --use_logits_processor True --schema_type squad_multi")
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

echo "Finished running in $(($(date +%s) - $start_time)) seconds"

# python3 src/eval_squad.py.py --data_path $FOLDER 

# python3 src/squad/eval_squad_multi.py --data_path $FOLDER