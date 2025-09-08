#!/bin/bash
set -e

FOLDER=${1:-"data/enforce_only_encoding"}

NUM_JOBS_PARALLEL=$(nvidia-smi -L  | wc -l)

# Run prompt experiments
# MODELS=("code-llama" "code-llama-instruct")
MODELS=("llama2" "llama2-chat" "llama3" "mistral" "mistral-instruct" "phi3")
TYPES=(unstructured)
# TYPES=("unstructured")
# # TYPES=(structured)
PROMPTS=(1 5 8)
SEEDS=(42)
# K_SHOT=(0 1 2 4 6 8 9)
K_SHOT=(9)

# Make this smarter split up creation

INPUT_FILES_UNSTRUCTURED=()

# # --max_examples 1000 \
echo "Creating Input Files"
for SEED in ${SEEDS[@]}; do
    for PROMPT in ${PROMPTS[@]}; do
        for TYPE in ${TYPES[@]}; do
            for K in ${K_SHOT[@]}; do
                INPUT_FILE=${FOLDER}/${SEED}/input/${TYPE}_${PROMPT}_${K}.pkl
                # WHen file not exist create it
                if [ ! -f $INPUT_FILE ]; then
                    python3 src/prepare_data.py \
                        --output $INPUT_FILE \
                        --template "${TYPE}_${PROMPT}" \
                        --shuffle True \
                        --random_seed $SEED \
                        --split "validation" \
                        --k_shot $K
                fi
                INPUT_FILES_UNSTRUCTURED+=($INPUT_FILE)
            done
        done
    done
done

echo "Created ${#INPUT_FILES_UNSTRUCTURED[@]} Unstructured Files"


# commands=()
# for MODEL in ${MODELS[@]}; do
#     OUTPUT_FILES_STRUCTURED=()
#     for INPUT_FILE in ${INPUT_FILES_STRUCTURED[@]}; do
#         OUTPUT_FILES_STRUCTURED=${INPUT_FILE//input\//output\/${MODEL}_}
#         OUTPUT_FILES_STRUCTURED+=($OUTPUT_FILES_STRUCTURED)
#     done
#     commands+=("python3 src/generate.py --input_files ${INPUT_FILES_STRUCTURED[@]} --output_files ${OUTPUT_FILES_STRUCTURED[@]} --model $MODEL --use_logits_processor True")

#     OUTPUT_FILES_UNSTRUCTURED=()
#     for INPUT_FILE in ${INPUT_FILES_UNSTRUCTURED[@]}; do
#         OUTPUT_FILES_UNSTRUCTURED=${INPUT_FILE//input\//output\/${MODEL}_}
#         OUTPUT_FILES_UNSTRUCTURED+=($OUTPUT_FILES_UNSTRUCTURED)
#     done
#     commands+=("python3 src/generate.py --input_files ${INPUT_FILES_UNSTRUCTURED[@]} --output_files ${OUTPUT_FILES_UNSTRUCTURED[@]} --model $MODEL --use_logits_processor False")
# done

commands=()
for MODEL in ${MODELS[@]}; do
    # Create command for unstructured files
    INPUT_FILES_STRING=""
    OUTPUT_FILES_STRING=""
    for INPUT_FILE in ${INPUT_FILES_UNSTRUCTURED[@]}; do
        OUTPUT_FILE=${INPUT_FILE//input\//output\/${MODEL}_}
        INPUT_FILES_STRING+="$INPUT_FILE "
        OUTPUT_FILES_STRING+="$OUTPUT_FILE "
    done
    commands+=("src/generate.py --input_files $INPUT_FILES_STRING --output_files $OUTPUT_FILES_STRING --model $MODEL --use_logits_processor True")
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
    CUDA_VISIBLE_DEVICES=$gpu python3 $cmd &> logs/$num.txt
    sleep 2
}

export -f run_with_parallel

echo "Running ${#commands[@]} commands with $NUM_JOBS_PARALLEL parallel jobs"

# echo "Commands: ${commands[0]}"

# echo "Commands: ${commands[1]}"

start_time=$(date +%s)

parallel -j $NUM_JOBS_PARALLEL --ungroup -k run_with_parallel {} {%} {#} ::: "${commands[@]}"

echo "Finished running in $(($(date +%s) - $start_time)) seconds"
