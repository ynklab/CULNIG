## Neuron-Level Analysis of Cultural Understanding in Large Language Models

![pipeline](/assets/CULNIG_pipeline.jpg)

This repository provides code and datasets for the paper "Neuron-Level Analysis of Cultural Understanding in Large Language Models". We implement CULNIG (Culture Neuron Identification Pipeline with Gradient-based Scoring), a method to identify culture-general and culture-specific neurons in LLMs via gradient-based scoring.


## Setup

- Install uv (see: https://docs.astral.sh/uv/getting-started/installation/)
- Create a virtual environment and install dependencies:
```bash
uv venv
uv sync
```
- Prepare datasets used by `dataset.py`:
    - BLEnD: Download `US_questions.csv` from the [BLEnD repo](https://github.com/nlee0212/BLEnD) and place it under the `data/BLEnD/` directory (e.g., `data/BLEnD/US_questions.csv`).
    - WorldValuesBench: Follow the instructions in the [WorldValuesBench repo](https://github.com/Demon702/WorldValuesBench), then place `question_metadata.json`, `full_demographic_qa.tsv`, and `full_value_qa.tsv` under `data/WorldValuesBench/`.


## CULNIG Pipeline

- `CULNIG/calc_neuron_score.py`: Compute neuron scores on a target dataset using gradient-based scoring.
    - Example:
        ```bash
        uv run python CULNIG/calc_neuron_score.py --model_name <model_name> --dataset_names blend
        ```
    - Available models: `google/gemma-3-12b-it`, `google/gemma-3-27b-it`, `Qwen/Qwen3-14B`, `meta-llama/Llama-3.1-8B-Instruct`, `microsoft/phi-4`, `tiiuae/Falcon3-10B-Instruct`
        - You can add other models by extending the code.
    - Through the pipeline, compute neuron scores for both `blend` and `blendcontrol`.
    - Neuron scores for CountryRC are computed every time.

- `CULNIG/decide_culture_general_neurons.py`: Identify culture-general neurons based on computed scores.
    - Example:
        ```bash
        uv run python CULNIG/decide_culture_general_neurons.py --model_name <model_name> --dataset_names blend
        ```

- `CULNIG/decide_culture_specific_neuron.py`: Identify culture-specific neurons based on computed scores.
    - Example:
        ```bash
        uv run python CULNIG/decide_culture_specific_neuron.py --model_name <model_name> --dataset_names blend
        ```

- `CULNIG/decide_random_neuron.py`: Select random neurons as a baseline.
    - Example:
        ```bash
        uv run python CULNIG/decide_random_neuron.py --model_name <model_name> --mlp_neuron_num <num> --attention_neuron_num 0
        ```
    - Note: This script currently does not treat MLP and attention neurons separately; attention neurons are included in MLP neurons (to match CULNIG). You can modify the code to separate them if desired.
    - Run `CULNIG/calc_neuron_score.py` beforehand to populate model architecture info used here.


## Evaluation

- `eval/evaluate.py`: Evaluate a model on a dataset, with optional neuron manipulation.
    - Example:
        ```bash
        uv run python eval/evaluate.py --model_name <model_name> --dataset_name <dataset_name> --neuron_file <path_to_neuron_file> --operation suppress
        ```
    - To evaluate without neuron manipulation, set `--neuron_file None`.
    - Available datasets: `blend`, `culturalbench`, `normad`, `worldvaluesbench`, `countryrc`, `commonsenseqa`, `qnli`, `mrpc`
    - Operations: `suppress`, `enhance`
    - Results are saved under `outputs/`.
    - For BLEnD, the script evaluates all questions (both BLEnD_neur and BLEnD_test). To evaluate only BLEnD_test, modify the code to load test questions only (`target_data='all' -> 'non_neuron'`).


## Fine-tuning

- `train/train.py`: Fine-tune a model on a dataset by updating specified modules.
    - Example:
        ```bash
        uv run python train/train.py --config train/config.yaml
        ```
    - An example config is provided at `train/config.yaml`; edit as needed.
    - To monitor with Weights & Biases, set environment variables beforehand:
        ```bash
        export WANDB_API_KEY=<your_wandb_api_key>
        export WANDB_PROJECT=<your_wandb_project_name>
        ```
    - Trained models and training logs are saved in `model_outputs/` (or the directory configured in your settings).


## Notes

- Issues and contributions are welcome via GitHub Issues and PRs.
