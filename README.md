# Neuron-Level Analysis of Cultural Understanding in Large Language Models

This repository contains code and datasets for the paper "Neuron-Level Analysis of Cultural Understanding in Large Language Models".

We introduce a method to identify culture-general and culture-specific neurons in LLMs, called CULNIG (Culture Neuron Identification Pipeline with Gradient-based Scoring).


## Setup

- Install uv (reference: https://docs.astral.sh/uv/getting-started/installation/)
- Install dependencies:
```bash
$ uv venv
$ uv sync
```
- Prepares datasets for identifying neurons and evaluating models (used in `dataset.py`)
  - For BLEnD, users must download the question file (`US_questions.csv`) from [repo](https://github.com/nlee0212/BLEnD) and place it in the `data` directory.
  - For WorldValuesBench, users must generate the dataset based on the instructions in the [repo](https://github.com/Demon702/WorldValuesBench) and place the generated files (`question_metadata.json`, `full_demographic_qa.tsv`, and `full_value_qa.tsv`) in the `data` directory.

## Scripts

### CULNIG

- `CULNIG/calc_neuron_scores.py`: Calculate neuron scores on a specified dataset using gradient-based scoring.
    - `$ uv run python CULNIG/calc_neuron_scores.py --model_name <model_name> --dataset_names blend`
    - available models: `google/gemma-3-12b-it`, `google/gemma-3-27b-it`, `Qwen/Qwen3-14B`, `meta-llama/Llama-3.1-8B-Instruct`, `microsoft/phi-4`, `tiiuae/Falcon3-10B-Instruct`
        - you can add other models by modifying the code.
    - note that for the CULNIG method, you need to calculate neuron scores on `blend` and `blendcontrol` datasets
    - the script calculates neuron scores for CountryRC every time
- `CULNIG/decide_culture_general_neurons.py`: Identify culture-general neurons based on the calculated neuron scores.
    - `$ uv run python CULNIG/decide_culture_general_neurons.py --model_name <model_name> --dataset_names blend --method max`
- `CULNIG/decide_culture_specific_neuron.py`: Identify culture-specific neurons based on the calculated neuron scores.
    - `$ uv run python CULNIG/decide_culture_specific_neuron.py --model_name <model_name> --dataset_names blend --method max`
- `CULNIG/decide_random_neuron.py`: Identify random neurons as a baseline.
    - `$ uv run python CULNIG/decide_random_neuron.py --model_name <model_name> --mlp_neuron_num <num> --attention_neuron_num 0`
    - currently, this script does not treat MLP and attention neurons separately and attention neurons are included in MLP neurons (to match CULNIG). You can modify the code if you want to treat them separately.

### Evaluation

- `eval/evaluate.py`: Evaluate the model on a specified dataset with optional neuron manipulation.
    - `$ uv run python eval/evaluate.py --model_name <model_name> --dataset_name <dataset_name> --neuron_file <path_to_neuron_file> --operation suppress`
    - if you want to evaluate without neuron manipulation, set `--neuron_file` to `None`
    - available datasets: `blend`, `culturalbench`, `normad`, `worldvaluesbench`, `countryrc`, `commonsenseqa`, `qnli`, `mrpc`
    - available operations: `suppress`, `enhance`
    - the script saves the evaluation results in the `outputs` directory

### Fine-tuning

- `train/train.py`: Fine-tune the model on a specified dataset, by updating specified modules.
    - `$ uv run python train/train.py --config <path_to_config_file>`
    - example config file is provided as `train/config.yaml`
    - you can modify the config files to change the training settings
