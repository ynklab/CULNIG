## Neuron-Level Analysis of Cultural Understanding in Large Language Models ([Paper](https://arxiv.org/abs/2510.08284))

![pipeline](/assets/CULNIG_pipeline.jpg)

This repository provides code and datasets for the paper "Neuron-Level Analysis of Cultural Understanding in Large Language Models". We implement **CULNIG** (Culture Neuron Identification Pipeline with Gradient-based Scoring), a method to identify _culture-general_ and _culture-specific_ neurons in LLMs via gradient-based scoring.


## Setup

- Install uv (see: https://docs.astral.sh/uv/getting-started/installation/)
- Create a virtual environment and install dependencies:
    - Adjust the `pyproject.toml` file to meet your environment if necessary.
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
    - Run this script after `CULNIG/calc_neuron_score.py` on both `blend` and `blendcontrol` as this script uses scores from both datasets to identify _culture-general_ neurons.

- `CULNIG/decide_culture_specific_neuron.py`: Identify culture-specific neurons based on computed scores.
    - Example:
        ```bash
        uv run python CULNIG/decide_culture_specific_neuron.py --model_name <model_name> --dataset_names blend
        ```
    - Run this script after `CULNIG/calc_neuron_score.py` on both `blend` and `blendcontrol` as this script uses scores from both datasets to identify _culture-specific_ neurons.

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
    - To evaluate without neuron manipulation, unset `--neuron_file`.
    - Available datasets: `blend`, `culturalbench`, `normad`, `worldvaluesbench`, `countryrc`, `commonsenseqa`, `qnli`, `mrpc`
    - Operations: `suppress`, `enhance`
    - Results are saved under `outputs/`.
    - For BLEnD, the script evaluates all questions (both BLEnD_neur and BLEnD_test). To evaluate only BLEnD_test, modify the code to load test questions only (`target_data='all' -> 'non_neuron'`).
- `eval/evaluate_blend_saq.py`: Evaluate a model on BLEnD SAQs with optional neuron manipulation.
    - Example:
        ```bash
        uv run python eval/evaluate_blend_saq.py --model_name <model_name> --neuron_file <path_to_neuron_file> --operation suppress
        ```
    - To evaluate without neuron manipulation, unset `--neuron_file`.
    - Operations: `suppress`, `enhance`
    - Results are saved under `outputs/blend_sqa`
    - For judging correctness, we use lemmatizers/stemmers/tokenizers of each language, following the original BLEnD paper and [repo](https://github.com/nlee0212/BLEnD/tree/9972379c4fd20601691c45e6d7befa6a3eed7ed4). We place the codes for our evaluation in `eval/lemmatizers/`. You can use the scripts as `uv run python lemma.py --input_file <input_file>`. For the detailed usage of each script, please refer to the comments in the code.


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

## Citation

If you find this work useful, please cite our paper:

```
@inproceedings{yamamoto2026neuronlevel,
title={Neuron-Level Analysis of Cultural Understanding in Large Language Models},
author={Taisei Yamamoto and Ryoma Kumon and Danushka Bollegala and Hitomi Yanaka},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=HZMmM3Dmri}
}
```

## Contact

For any questions or inquiries, please contact: yamamo96[at]is.s.u-tokyo.ac.jp

## Notes

- Issues and contributions are welcome via GitHub Issues and PRs.
