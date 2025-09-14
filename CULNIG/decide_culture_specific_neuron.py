import argparse
import logging
import json
from pathlib import Path
from collections import defaultdict


MLP_COUNTRY_NEURON_PROPORTION = 0.003   # top t% of neurons for each country for MLP
MLP_COUNTRYRC_NEURON_PROPORTION = 0.01   # top r% neurons on CountryRC to exclude for MLP
ATTENTION_COUNTRY_NEURON_PROPORTION = 0.00  # top t% of neurons for each country for Attention
ATTENTION_COUNTRYRC_NEURON_PROPORTION = 0.0  # top r% neurons on CountryRC to exclude for Attention
COUNTRY_SCORE_ZSCORE_THRESHOLD = 0.5  # threshold for z-score of country score
OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'outputs'
TARGET_COUNTRIES =  [
    'China', 'Indonesia', 'Iran', 'Mexico', 'South Korea', 'Spain', 'UK', 'USA',
]
MLP_TARGET_MODULES = ['mlp.gate_proj', 'self_attn.v_proj', 'self_attn.q_proj', 'self_attn.k_proj']   # treat both MLP and Attention modules as same
ATTENTION_TARGET_MODULES = []


def parse_args():
    parser = argparse.ArgumentParser(description="Decide culture-specific neuron.")
    parser.add_argument('--model_name', type=str, required=True, help='Name of the pre-trained model to use.')
    parser.add_argument('--dataset_names', nargs='+', required=True, help='Names of the datasets to use. Can be a list of dataset names.')
    parser.add_argument('--method', type=str, required=True, help='Name of the method for neuron identification.')
    return parser.parse_args()


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    logger = logging.getLogger(__name__)
    return logger


def main():
    args = parse_args()
    logger = setup_logging()
    dataset_names = args.dataset_names
    dataset_names.sort()  # Sort dataset names for consistent file naming
    method = args.method
    if method:
        method_name = '_' + method
    else:
        method_name = ''
    logger.info(f"Deciding cultural neurons for model: {args.model_name} on dataset: {dataset_names} using method: {method}")
    logger.info(f'Hyper parameters: MLP_COUNTRY_NEURON_PROPORTION={MLP_COUNTRY_NEURON_PROPORTION}, MLP_COUNTRYRC_NEURON_PROPORTION={MLP_COUNTRYRC_NEURON_PROPORTION}')
    logger.info(f'Hyper parameters: ATTENTION_COUNTRY_NEURON_PROPORTION={ATTENTION_COUNTRY_NEURON_PROPORTION}, ATTENTION_COUNTRYRC_NEURON_PROPORTION={ATTENTION_COUNTRYRC_NEURON_PROPORTION}')
    logger.info(f'Target modules: MLP_TARGET_MODULES={MLP_TARGET_MODULES}, ATTENTION_TARGET_MODULES={ATTENTION_TARGET_MODULES}')

    # 1. Load neuron scores for each dataset and compute overall neuron scores by country
    mlp_neuron_scores = defaultdict(lambda: defaultdict(float))  # neuron_scores[f'{module_name}_{layer_idx}_{neuron_idx}'][country] = score
    attention_neuron_scores = defaultdict(lambda: defaultdict(float))  # neuron_scores[f'{module_name}_{layer_idx}_{neuron_idx}'][country] = score
    dataset_ids = defaultdict(list)
    for dataset_name in dataset_names:
        score_path = OUTPUT_DIR / args.model_name.split('/')[-1] / 'cultural_neuron' / f'{dataset_name}{method_name}_scores.json'
        control_score_path = OUTPUT_DIR / args.model_name.split('/')[-1] / 'cultural_neuron' / f'{dataset_name}control{method_name}_scores.json'
        if not score_path.exists():
            raise FileNotFoundError(f"Score file not found: {score_path}")
        if not control_score_path.exists():
            raise FileNotFoundError(f"Control score file not found: {control_score_path}")

        with open(score_path, 'r') as f:
            scores_dict = json.load(f)
        with open(control_score_path, 'r') as f:
            control_scores_dict = json.load(f)
        _dataset_neuron_scores = scores_dict['neuron_scores']  # dataset_neuron_scores[f'{module_name}_{layer_idx}_{neuron_idx}'] = {country: score}
        _control_dataset_neuron_scores = control_scores_dict['neuron_scores']
        dataset_sample_num = len(scores_dict['dataset_ids'][dataset_name])
        control_sample_num = len(control_scores_dict['dataset_ids'][f'{dataset_name}control'])

        # calculate dataset-level neuron scores by subtracting control scores
        mlp_dataset_neuron_scores = defaultdict(float)  # dataset_neuron_scores[f'{module_name}_{layer_idx}_{neuron_idx}'] = score
        attention_dataset_neuron_scores = defaultdict(float)  # dataset_neuron_scores[f'{module_name}_{layer_idx}_{neuron_idx}'] = score
        for key, score in _dataset_neuron_scores.items():
            parts = key.split('_')
            module_name = '_'.join(parts[:-2])
            if module_name in MLP_TARGET_MODULES:
                dataset_score = sum(score.values()) / dataset_sample_num   # divide by number of datasets to normalize across datasets
                control_score = sum(_control_dataset_neuron_scores[key].values()) / control_sample_num
                mlp_dataset_neuron_scores[key] = dataset_score - control_score
                for country, value in score.items():
                    # normalize by country probability
                    dataset_val = value / dataset_sample_num
                    control_val = _control_dataset_neuron_scores[key][country] / control_sample_num
                    mlp_neuron_scores[key][country] += dataset_val - control_val
            elif module_name in ATTENTION_TARGET_MODULES:
                dataset_score = sum(score.values()) / dataset_sample_num   # divide by number of datasets to normalize across datasets
                control_score = sum(_control_dataset_neuron_scores[key].values()) / control_sample_num
                attention_dataset_neuron_scores[key] = dataset_score - control_score
                for country, value in score.items():
                    dataset_val = value / dataset_sample_num
                    control_val = _control_dataset_neuron_scores[key][country] / control_sample_num
                    attention_neuron_scores[key][country] += dataset_val - control_val

        dataset_dataset_ids = scores_dict['dataset_ids']  # dataset_ids[dataset_name] = [id1, id2, ...]
        for dname, ids in dataset_dataset_ids.items():
            dataset_ids[dname].extend(ids)

    # 2. Load CountryRC neuron scores
    countryrc_score_path = OUTPUT_DIR / args.model_name.split('/')[-1] / 'cultural_neuron' / f'countryrc{method_name}_scores.json'
    if not countryrc_score_path.exists():
        raise FileNotFoundError(f"CountryRC score file not found: {countryrc_score_path}")
    with open(countryrc_score_path, 'r') as f:
        countryrc_scores_dict = json.load(f)
    countryrc_neuron_scores = countryrc_scores_dict['neuron_scores']   # countryrc_neuron_scores[f'{module_name}_{layer_idx}_{neuron_idx}'] = {country: score}
    # dataset_idsの収集
    countryrc_dataset_ids = countryrc_scores_dict['dataset_ids']
    for dataset_name, ids in countryrc_dataset_ids.items():
        dataset_ids[dataset_name].extend(ids)

    # 3. Calculate country-specific neuron scores by selecting top t% neurons for each country, excluding top r% CountryRC neurons for each country
    for country in TARGET_COUNTRIES:
        mlp_country_neuron_scores = {}
        attention_country_neuron_scores = {}
        for key, score in mlp_neuron_scores.items():
            parts = key.split('_')
            module_name = '_'.join(parts[:-2])
            assert country in score, f"Country {country} not found in scores for neuron {key}"
            mlp_country_neuron_scores[key] = score[country]
        for key, score in attention_neuron_scores.items():
            parts = key.split('_')
            module_name = '_'.join(parts[:-2])
            assert country in score, f"Country {country} not found in scores for neuron {key}"
            attention_country_neuron_scores[key] = score[country]
        mlp_country_neuron_all_scores = {k: v for k, v in sorted(mlp_country_neuron_scores.items(), key=lambda item: item[1], reverse=True)}
        mlp_country_neuron_count = int(len(mlp_country_neuron_all_scores) * MLP_COUNTRY_NEURON_PROPORTION)
        mlp_country_neurons = list(mlp_country_neuron_all_scores.keys())[:mlp_country_neuron_count]
        attention_country_neuron_all_scores = {k: v for k, v in sorted(attention_country_neuron_scores.items(), key=lambda item: item[1], reverse=True)}
        attention_country_neuron_count = int(len(attention_country_neuron_all_scores) * ATTENTION_COUNTRY_NEURON_PROPORTION)
        attention_country_neurons = list(attention_country_neuron_all_scores.keys())[:attention_country_neuron_count]

        # fetch CountryRC neurons to exclude
        mlp_country_countryrc_scores = {}
        attention_country_countryrc_scores = {}
        for key, score in countryrc_neuron_scores.items():
            parts = key.split('_')
            module_name = '_'.join(parts[:-2])
            if module_name in MLP_TARGET_MODULES:
                mlp_country_countryrc_scores[key] = score[country]
            elif module_name in ATTENTION_TARGET_MODULES:
                attention_country_countryrc_scores[key] = score[country]

        mlp_country_countryrc_scores_sorted = {k: v for k, v in sorted(mlp_country_countryrc_scores.items(), key=lambda item: item[1], reverse=True)}
        attention_country_countryrc_scores_sorted = {k: v for k, v in sorted(attention_country_countryrc_scores.items(), key=lambda item: item[1], reverse=True)}
        mlp_country_countryrc_count = int(len(mlp_country_countryrc_scores_sorted) * MLP_COUNTRYRC_NEURON_PROPORTION)
        attention_country_countryrc_count = int(len(attention_country_countryrc_scores_sorted) * ATTENTION_COUNTRYRC_NEURON_PROPORTION)
        mlp_countryrc_neurons = set(list(mlp_country_countryrc_scores_sorted.keys())[:mlp_country_countryrc_count])
        attention_countryrc_neurons = set(list(attention_country_countryrc_scores_sorted.keys())[:attention_country_countryrc_count])

        cultural_neurons = []
        scores_sum = 0.0
        other_scores_sum = 0.0
        module_count = defaultdict(int)  # count of neurons per module
        for key in mlp_country_neurons:
            if key in mlp_countryrc_neurons:
                continue
            # calculate z-score of the country score
            scores = mlp_neuron_scores[key]
            scores_mean = sum(scores.values()) / len(scores)
            scores_std = (sum((x - scores_mean) ** 2 for x in scores.values()) / len(scores)) ** 0.5
            if scores_std == 0:
                z_score = 0
            else:
                z_score = (mlp_country_neuron_scores[key] - scores_mean) / scores_std
            if z_score < COUNTRY_SCORE_ZSCORE_THRESHOLD:
                continue

            parts = key.split('_')
            neuron_idx = int(parts[-1])
            layer_idx = int(parts[-2])
            module_name = '_'.join(parts[:-2])
            cultural_neurons.append({
                'module_name': module_name,
                'layer_idx': layer_idx,
                'neuron_idx': neuron_idx,
                'attribute_score': mlp_country_neuron_scores[key],
                'scores': mlp_neuron_scores[key],
            })
            scores_sum += mlp_country_neuron_scores[key]
            other_scores_sum += sum(mlp_neuron_scores[key].values()) - mlp_country_neuron_scores[key]
            module_count[module_name] += 1
        for key in attention_country_neurons:
            if key in attention_countryrc_neurons:
                continue
            # calculate z-score of the country score
            scores = attention_neuron_scores[key]
            scores_mean = sum(scores.values()) / len(scores)
            scores_std = (sum((x - scores_mean) ** 2 for x in scores.values()) / len(scores)) ** 0.5
            if scores_std == 0:
                z_score = 0
            else:
                z_score = (attention_country_neuron_scores[key] - scores_mean) / scores_std
            if z_score < COUNTRY_SCORE_ZSCORE_THRESHOLD:
                continue

            parts = key.split('_')
            neuron_idx = int(parts[-1])
            layer_idx = int(parts[-2])
            module_name = '_'.join(parts[:-2])
            cultural_neurons.append({
                'module_name': module_name,
                'layer_idx': layer_idx,
                'neuron_idx': neuron_idx,
                'attribute_score': attention_country_neuron_scores[key],
                'scores': attention_neuron_scores[key],
            })
            scores_sum += attention_country_neuron_scores[key]
            other_scores_sum += sum(attention_neuron_scores[key].values()) - attention_country_neuron_scores[key]
            module_count[module_name] += 1
        logger.info(f"Module count for country {country}: {dict(module_count)}")
        logger.info(f"Scores sum: {scores_sum}, Other scores sum: {other_scores_sum}")
        # save the results
        result = {
            'model_name': args.model_name,
            'dataset_ids': dataset_ids,
            'top_neurons': cultural_neurons,
        }
        output_path = OUTPUT_DIR / args.model_name.split('/')[-1] / 'cultural_neuron' / f'{country.replace(' ', '')}_neurons_{''.join(dataset_names)}{method_name}.json'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=4)
        logger.info(f"Saved cultural neurons for {country} to {output_path}, {len(cultural_neurons)} neurons found.")


if __name__ == "__main__":
    main()
