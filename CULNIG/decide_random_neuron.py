import argparse
import logging
import json
from pathlib import Path
import random
from collections import defaultdict


OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'outputs'
MLP_TARGET_MODULES = ['mlp.gate_proj', 'self_attn.v_proj', 'self_attn.q_proj', 'self_attn.k_proj']
ATTENTION_TARGET_MODULES = []


def parse_args():
    parser = argparse.ArgumentParser(description="Select random neuron.")
    parser.add_argument('--model_name', type=str, required=True, help='Name of the pre-trained model to use.')
    parser.add_argument('--mlp_neuron_num', type=int, required=True, help='Number of MLP neurons to select.')
    parser.add_argument('--attention_neuron_num', type=int, required=True, help='Number of attention neurons to select.')
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
    mlp_neuron_num = args.mlp_neuron_num
    attention_neuron_num = args.attention_neuron_num
    logger.info(f"Deciding random neurons for model: {args.model_name}")
    logger.info(f'Hyper parameters: MLP_NEURON_NUM={mlp_neuron_num}, ATTENTION_NEURON_NUM={attention_neuron_num}')
    logger.info(f'Target modules: MLP_TARGET_MODULES={MLP_TARGET_MODULES}, ATTENTION_TARGET_MODULES={ATTENTION_TARGET_MODULES}')

    # candidate neuronsを取得
    ref_path = OUTPUT_DIR / args.model_name.split('/')[-1] / 'cultural_neuron' / f'blend_max_scores.json'
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference scores file not found: {ref_path}")
    with open(ref_path, 'r') as f:
        ref_data = json.load(f)
    ref_neuron_scores = ref_data['neuron_scores']  # neuron_scores[f'{module_name}_{layer_idx}_{neuron_idx}'] = {country: score}

    mlp_candidate_neurons = []
    attention_candidate_neurons = []
    module_count = defaultdict(int)
    for key in ref_neuron_scores.keys():
        parts = key.split('_')
        module_name = '_'.join(parts[:-2])
        layer_idx = int(parts[-2])
        neuron_idx = int(parts[-1])
        if module_name in MLP_TARGET_MODULES:
            mlp_candidate_neurons.append((module_name, layer_idx, neuron_idx))
            module_count[module_name] += 1
        elif module_name in ATTENTION_TARGET_MODULES:
            attention_candidate_neurons.append((module_name, layer_idx, neuron_idx))
            module_count[module_name] += 1
    logger.info(f'neuron counts for all modules: {dict(module_count)}')

    # select random neurons with 10 seeds
    for seed in range(10):
        random.seed(seed)
        random_neurons = []
        module_count = defaultdict(int)
        if mlp_neuron_num > 0:
            mlp_random_neurons = random.sample(mlp_candidate_neurons, mlp_neuron_num)
            for module_name, layer_idx, neuron_idx in mlp_random_neurons:
                random_neurons.append({
                    'module_name': module_name,
                    'layer_idx': layer_idx,
                    'neuron_idx': neuron_idx,
                    'attribute_score': 0.0,  # since randomly selected, score is 0.0
                })
                module_count[module_name] += 1
        if attention_neuron_num > 0:
            attention_random_neurons = random.sample(attention_candidate_neurons, attention_neuron_num)
            for module_name, layer_idx, neuron_idx in attention_random_neurons:
                random_neurons.append({
                    'module_name': module_name,
                    'layer_idx': layer_idx,
                    'neuron_idx': neuron_idx,
                    'attribute_score': 0.0,  # since randomly selected, score is 0.0
                })
                module_count[module_name] += 1

        dataset_ids = ref_data['dataset_ids']

        # save the results
        result = {
            'model_name': args.model_name,
            'dataset_ids': dataset_ids,
            'top_neurons': random_neurons,
        }
        output_path = OUTPUT_DIR / args.model_name.split('/')[-1] / 'cultural_neuron' / f'all_neurons_random{seed}.json'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=4)
        logger.info(f"Saved random neurons to {output_path}, {len(random_neurons)} neurons found.")
        logger.info(f"Module counts: {dict(module_count)}")


if __name__ == "__main__":
    main()
