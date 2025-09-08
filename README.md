# The Hidden Cost of Structure: How Constrained Decoding Affects Language Model Performance

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![GPU Required](https://img.shields.io/badge/GPU-Required-green.svg)](https://developer.nvidia.com/cuda-gpus)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Abstract

This repository contains the code and experiments for evaluating how constrained decoding methods affect the performance of large language models across multiple benchmarks. Our research investigates the trade-offs between structural constraints and model performance on tasks like question answering, instruction following, and factual accuracy.

## Key Findings

- Constrained decoding can significantly impact model performance depending on the task
- Trade-offs exist between output structure compliance and answer quality
- Different constraint types affect various model capabilities differently

## Requirements
- Docker with NVIDIA Container Toolkit


## Setup

### Option 1: Docker (Recommended)

1. **Build the Docker image:**
   ```bash
   docker build -t constrained .
   ```

2. **Run the container with GPU support:**
   ```bash
        scripts/run-in-docker.sh -g 0
   ```

### Option 2: Local Installation

1. **Create conda environment:**
   ```bash
   conda env create -f environment.yml
   conda activate constrained-decoding
   ```


## Running Experiments

### Quick Start

Once inside the Docker container or with the environment activated run the script for your experiment e.g.:

1. **Run SQuAD experiments:**
   ```bash
   ./scripts/run_squad.sh
   ```

2. **Run MMLU benchmark:**
   ```bash
   ./scripts/run_mmlu.sh
   ```

3. **Run instruction following evaluation:**
   ```bash
   ./scripts/run_ifeval.sh
   ```





## Evaluation and Analysis

### Processing Results

1. **Convert FactScore results:**
   ```bash
   jupyter notebook src/convert_factscore.ipynb
   ```

2. **Analyze constraint compliance:**
   ```bash
   python src/llm_as_judge.py --results_dir results/
   ```


## Citation

If you use this code or findings in your research, please cite:

```bibtex
@article{schall2025hidden,
TODO
}
```

## Troubleshooting


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


