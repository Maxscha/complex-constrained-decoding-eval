#!/bin/bash -e

# The hardcoded Python script path
PYTHON_SCRIPT="src/llm_as_judge.py"
# Check if directory is provided
if [ $# -eq 0 ]; then
    search_dir="."
else
    search_dir="$1"
fi

# Verify search directory exists
if [ ! -d "$search_dir" ]; then
    echo "Error: Directory '$search_dir' does not exist."
    exit 1
fi

echo "Searching for files in: $search_dir"

# Find all output files containing 'unstructured'
find "$search_dir" -type f -path "*/output/*" -name "*structured*.pkl" | while read -r output_file; do
    # Extract the 'unstructured_X_X.pkl' part and construct input path
    base_name=$(basename "$output_file" | sed -E 's/.*_(structured_[0-9]+\.pkl)$/\1/')
    input_file=$(dirname "$(dirname "$output_file")")/input/$base_name
    
    # Verify input file exists
    if [ ! -f "$input_file" ]; then
        echo "Warning: Input file not found: $input_file"
        continue
    fi
    
    echo "Processing:"
    echo "Input:  $input_file"
    echo "Output: $output_file"
    
    # Run Python script with the paths
    python $PYTHON_SCRIPT --input_path "$input_file" --output_path "$output_file" --multi
    
    if [ $? -eq 0 ]; then
        echo "Successfully processed files."
    else
        echo "Error: Processing failed"
    fi
    echo "-------------------"
done