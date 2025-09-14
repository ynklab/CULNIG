import argparse
import logging
import json
from pathlib import Path
from collections import defaultdict


MLP_CULTURE_NEURON_PROPORTION = 0.01   # top t% neurons on BLEnD scores for MLPs
MLP_COUNTRYRC_NEURON_PROPORTION = 0.01   # top r% neurons on CountryRC scores for MLPs
ATTENTION_CULTURE_NEURON_PROPORTION = 0.002  # top t% neurons on BLEnD scores for Attention
ATTENTION_COUNTRYRC_NEURON_PROPORTION = 0.01  # top r% neurons on CountryRC scores for Attention
OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'outputs'
MLP_TARGET_MODULES = ['mlp.gate_proj']
ATTENTION_TARGET_MODULES = ['self_attn.v_proj', 'self_attn.q_proj', 'self_attn.k_proj']
MLP_SAVE_MODULES = ['mlp.gate_proj']
ATTENTION_SAVE_MODULES = ['self_attn.v_proj', 'self_attn.q_proj', 'self_attn.k_proj']


def parse_args():
    parser = argparse.ArgumentParser(description="Identify culture-general neurons based on pre-calculated scores.")
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to use.')
    parser.add_argument('--dataset_names', nargs='+', required=True, help='Names of the datasets to for scores. Can be a list of dataset names.')
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
    logger.info(f"Deciding cultural neurons for model: {args.model_name} on dataset: {dataset_names}")
    logger.info(f'Hyper parameters: MLP_CULTURE_NEURON_PROPORTION={MLP_CULTURE_NEURON_PROPORTION}, MLP_COUNTRYRC_NEURON_PROPORTION={MLP_COUNTRYRC_NEURON_PROPORTION}')
    logger.info(f'Target modules: MLP_TARGET_MODULES={MLP_TARGET_MODULES}, Save modules: MLP_SAVE_MODULES={MLP_SAVE_MODULES}')
    logger.info(f'Hyper parameters: ATTENTION_CULTURE_NEURON_PROPORTION={ATTENTION_CULTURE_NEURON_PROPORTION}, ATTENTION_COUNTRYRC_NEURON_PROPORTION={ATTENTION_COUNTRYRC_NEURON_PROPORTION}')
    logger.info(f'Target modules: ATTENTION_TARGET_MODULES={ATTENTION_TARGET_MODULES}, Save modules: ATTENTION_SAVE_MODULES={ATTENTION_SAVE_MODULES}')

    # 1. Sum neuron scores for each dataset, find the top neurons
    mlp_neuron_scores = defaultdict(float)  # neuron_scores[f'{module_name}_{layer_idx}_{neuron_idx}'] = score
    attention_neuron_scores = defaultdict(float)  # neuron_scores[f'{module_name}_{layer_idx}_{neuron_idx}'] = score
    dataset_ids = defaultdict(list)
    for dataset_name in dataset_names:
        score_path = OUTPUT_DIR / args.model_name.split('/')[-1] / 'cultural_neuron' / f'{dataset_name}_max_scores.json'
        control_score_path = OUTPUT_DIR / args.model_name.split('/')[-1] / 'cultural_neuron' / f'{dataset_name}control_max_scores.json'
        if not score_path.exists():
            raise FileNotFoundError(f"Score file not found: {score_path}")
        if not control_score_path.exists():
            raise FileNotFoundError(f"Control score file not found: {control_score_path}")
        with open(score_path, 'r') as f:
            scores_dict = json.load(f)
        with open(control_score_path, 'r') as f:
            control_scores_dict = json.load(f)

        dataset_dataset_ids = scores_dict['dataset_ids']  # dataset_ids[dataset_name] = [id1, id2, ...]
        for dname, ids in dataset_dataset_ids.items():
            # basically, dname should be the same as dataset_name, but just in case
            dataset_ids[dname].extend(ids)
        dataset_sample_num = len(dataset_dataset_ids[dataset_name])
        control_sample_num = len(control_scores_dict['dataset_ids'][f'{dataset_name}control'])

        dataset_neuron_scores = scores_dict['neuron_scores']  # dataset_neuron_scores[f'{module_name}_{layer_idx}_{neuron_idx}'] = {country: score}
        control_dataset_neuron_scores = control_scores_dict['neuron_scores']
        for key, score in dataset_neuron_scores.items():
            parts = key.split('_')
            module_name = '_'.join(parts[:-2])
            if module_name in MLP_TARGET_MODULES:
                # sum up MLP neuron scores
                dataset_score = sum(score.values()) / dataset_sample_num   # normalize by dataset sample count
                control_score = sum(control_dataset_neuron_scores[key].values()) / control_sample_num
                mlp_neuron_scores[key] += dataset_score - control_score   # subtract control score
            elif module_name in ATTENTION_TARGET_MODULES:
                # sum up Attention neuron scores
                dataset_score = sum(score.values()) / dataset_sample_num   # normalize by dataset sample count
                control_score = sum(control_dataset_neuron_scores[key].values()) / control_sample_num
                attention_neuron_scores[key] += dataset_score - control_score  # subtract control score

    # select top neurons based on the proportion t for each of MLP and Attention
    mlp_neuron_scores_sorted = sorted(mlp_neuron_scores.items(), key=lambda x: x[1], reverse=True)
    mlp_neuron_num = int(len(mlp_neuron_scores_sorted) * MLP_CULTURE_NEURON_PROPORTION)
    attention_neuron_scores_sorted = sorted(attention_neuron_scores.items(), key=lambda x: x[1], reverse=True)
    attention_neuron_num = int(len(attention_neuron_scores_sorted) * ATTENTION_CULTURE_NEURON_PROPORTION)
    mlp_culture_neurons = mlp_neuron_scores_sorted[:mlp_neuron_num]  # select top neurons for MLP
    attention_culture_neurons = attention_neuron_scores_sorted[:attention_neuron_num]  # select top neurons for Attention
    culture_neurons = mlp_culture_neurons + attention_culture_neurons  # integrate MLP and Attention cultural neurons
    logger.info(f"Selected {len(mlp_culture_neurons)} MLP cultural neurons based on the proportion {MLP_CULTURE_NEURON_PROPORTION}")
    logger.info(f"Selected {len(attention_culture_neurons)} Attention cultural neurons based on the proportion {ATTENTION_CULTURE_NEURON_PROPORTION}")

    # 2. Load CountryRC scores, find the top r% neurons
    mlp_countryrc_neuron_scores = defaultdict(float)  # countryrc_neuron_scores[f'{module_name}_{layer_idx}_{neuron_idx}'] = score
    attention_countryrc_neuron_scores = defaultdict(float)  # countryrc_neuron_scores[f'{module_name}_{layer_idx}_{neuron_idx}'] = score
    countryrc_score_path = OUTPUT_DIR / args.model_name.split('/')[-1] / 'cultural_neuron' / f'countryrc_max_scores.json'
    if not countryrc_score_path.exists():
        raise FileNotFoundError(f"CountryRC score file not found: {countryrc_score_path}")
    with open(countryrc_score_path, 'r') as f:
        countryrc_scores_dict = json.load(f)
    scores_dict = countryrc_scores_dict['neuron_scores']   # countryrc_neuron_scores[f'{module_name}_{layer_idx}_{neuron_idx}'] = {country: score}
    for key, score in scores_dict.items():
        parts = key.split('_')
        module_name = '_'.join(parts[:-2])
        if module_name in MLP_TARGET_MODULES:
            # sum up MLP neuron scores
            mlp_countryrc_neuron_scores[key] = sum(score.values())
        elif module_name in ATTENTION_TARGET_MODULES:
            # sum up Attention neuron scores
            attention_countryrc_neuron_scores[key] = sum(score.values())
    # collect dataset ids
    countryrc_dataset_ids = countryrc_scores_dict['dataset_ids']
    for dataset_name, ids in countryrc_dataset_ids.items():
        dataset_ids[dataset_name].extend(ids)

    # Select CountryRC neurons based on the proportion r for each of MLP and Attention
    mlp_countryrc_neuron_scores_sorted = sorted(mlp_countryrc_neuron_scores.items(), key=lambda x: x[1], reverse=True)
    attention_countryrc_neuron_scores_sorted = sorted(attention_countryrc_neuron_scores.items(), key=lambda x: x[1], reverse=True)
    mlp_countryrc_neuron_count = int(len(mlp_countryrc_neuron_scores_sorted) * MLP_COUNTRYRC_NEURON_PROPORTION)
    attention_countryrc_neuron_count = int(len(attention_countryrc_neuron_scores_sorted) * ATTENTION_COUNTRYRC_NEURON_PROPORTION)
    mlp_countryrc_neurons = set([neuron for neuron, score in mlp_countryrc_neuron_scores_sorted[:mlp_countryrc_neuron_count]])
    attention_countryrc_neurons = set([neuron for neuron, score in attention_countryrc_neuron_scores_sorted[:attention_countryrc_neuron_count]])
    logger.info(f"Selected {len(mlp_countryrc_neurons)} MLP CountryRC neurons based on the proportion {MLP_COUNTRYRC_NEURON_PROPORTION}")
    logger.info(f"Selected {len(attention_countryrc_neurons)} Attention CountryRC neurons based on the proportion {ATTENTION_COUNTRYRC_NEURON_PROPORTION}")
    countryrc_neurons = mlp_countryrc_neurons.union(attention_countryrc_neurons)  # Union of MLP and Attention CountryRC neurons

    # 3. Exclude CountryRC neurons from cultural neurons, save the results
    refined_culture_neurons = []
    module_count = defaultdict(int)  # Count neurons by module
    for neuron, score in culture_neurons:
        if neuron in countryrc_neurons:
            continue
        parts = neuron.split('_')
        neuron_idx = int(parts[-1])
        layer_idx = int(parts[-2])
        module_name = '_'.join(parts[:-2])
        # Only save neurons from specified modules
        if module_name not in MLP_SAVE_MODULES and module_name not in ATTENTION_SAVE_MODULES:
            continue
        refined_culture_neurons.append({
            'module_name': module_name,
            'layer_idx': layer_idx,
            'neuron_idx': neuron_idx,
            'attribute_score': score,
        })
        module_count[module_name] += 1

    logger.info(f"Refined cultural neurons count after excluding CountryRC neurons: {len(refined_culture_neurons)}")
    logger.info(f"Module counts: {dict(module_count)}")

    # 4. Save the results
    output_data = {
        'model_name': args.model_name,
        'dataset_ids': dataset_ids,
        'top_neurons': refined_culture_neurons,
    }
    output_path = OUTPUT_DIR / args.model_name.split('/')[-1] / 'cultural_neuron' / f'all_neurons_{''.join(dataset_names)}_max.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
