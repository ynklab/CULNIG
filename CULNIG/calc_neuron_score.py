import argparse
from collections import defaultdict
import logging
from pathlib import Path
import json
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import set_seed

# add python path to import dataset and utils
sys.path.append(str(Path(__file__).resolve().parent.parent))
from dataset import load_dataset_neuron_scores
from utils import get_text_model, get_target_module


DATA_ROOT = Path(__file__).resolve().parent.parent / 'data'
BATCH_SIZE = 16   # Batch size for processing samples
TARGET_COUNTRIES =  [
    'China', 'Indonesia', 'Iran', 'Mexico', 'South Korea', 'Spain', 'UK', 'USA',
]
TARGET_MODULES = ['mlp.gate_proj', 'mlp.up_proj', 'self_attn.v_proj', 'self_attn.q_proj', 'self_attn.k_proj']


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate neuron scores on a dataset.")
    parser.add_argument('--model_name', type=str, default='google/gemma-3-12b-it', help='Name of the LLM to use.')
    parser.add_argument('--dataset_names', nargs='+', default=['blend'], help='Names of the datasets to use. Can be a list of dataset names.')
    return parser.parse_args()


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    logger = logging.getLogger(__name__)
    return logger


def load_model(model_name='google/gemma-3-12b-it'):
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation='eager' if 'gemma-3' in model_name else 'sdpa',
    )   # eager attention is recommended for Gemma-3 models in transformers
    model = model.to('cuda')  # Move model to GPU
    return model, tokenizer


def calculate_scores(model, tokenizer, dataloader, logger):
    """Calculate neuron scores based on the provided instruction
    """
    # max aggregation methods
    max_neuron_scores = defaultdict(lambda: defaultdict(float))
    total_probabilities_per_country = defaultdict(float)

    activations = {}
    def save_activation(name):
        def hook(module, input, output):
            # Clone the output to ensure it maintains the gradient connection
            activations[name] = output
            output.retain_grad()  # Retain gradients for the output
        return hook

    hooks = []
    text_model = get_text_model(model)
    for i in range(len(text_model.layers)):
        for module_name in TARGET_MODULES:
            module = get_target_module(model, module_name, i)
            hooks.append(module.register_forward_hook(save_activation(f'model.model.layers.{i}.{module_name}')))

    # probability of the activation on every token for each neuron
    total_iter = len(dataloader)
    cur_iter = 0
    model.train()  # Set to training mode to enable gradient computation
    for batch in dataloader:
        cur_iter += 1
        if cur_iter % 100 == 0:
            logger.info(f"Processing batch {cur_iter}/{total_iter}")

        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        countries = batch['countries']
        dataset_names = batch['dataset_names']
        control = ['control' in dataset_name for dataset_name in dataset_names]

        model.zero_grad()  # Clear previous gradients
        activations.clear()  # Clear previous activations
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = output.logits[:, -1, :]  # Get logits for the last token
        probabilities = F.softmax(logits, dim=-1)  # Shape: (batch_size, vocab_size)

        # probabilities for the correct token
        labels = [str(label) for label in batch['labels']]
        labels_ids = torch.tensor([tokenizer.convert_tokens_to_ids(l) for l in labels], device=model.device)
        correct_probs = probabilities[torch.arange(probabilities.size(0)), labels_ids]  # Shape: (batch_size,)
        correct_probs.sum().backward()  # Backpropagate to compute gradients
        correct_probs_cpu_list = correct_probs.detach().cpu().tolist()  # Move to CPU and convert to list
        prob_total = correct_probs.unsqueeze(1)  # Shape: (batch_size, 1)
        # Calculate total probabilities for each country
        for country, prob in zip(countries, correct_probs_cpu_list):
            total_probabilities_per_country[country] += prob

        for module_name, activation in activations.items():
            # acvtivation: (batch_size, seq_len, num_neurons)
            parts = module_name.split('.')
            layer_idx = int(parts[3])  # Extract layer index from the module name
            mod_name = '.'.join(parts[4:])  # Get the module name after the layer index
            grads = activation.grad  # Get the gradients of the activations (Shape: (batch_size, seq_len, num_neurons))

            if model.name_or_path in ['google/gemma-3-12b-it', 'google/gemma-3-27b-it', 'Qwen/Qwen3-14B',
                                      'meta-llama/Llama-3.1-8B-Instruct', 'tiiuae/Falcon3-10B-Instruct']:
                pass
            elif model.name_or_path in ['microsoft/phi-4']:
                config = text_model.layers[layer_idx].config
                intermed_size = config.intermediate_size
                head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
                key_size = config.num_key_value_heads * head_dim
                query_size = config.num_attention_heads * head_dim
                if mod_name == 'mlp.gate_proj':
                    # first half of the output is gate_proj, second half is up_proj
                    activation = activation[:, :, :intermed_size]  # Keep only the gate_proj part
                    grads = grads[:, :, :intermed_size]  # Keep only the gate_proj gradients
                elif mod_name == 'mlp.up_proj':
                    activation = activation[:, :, intermed_size:]  # Keep only the up_proj part
                    grads = grads[:, :, intermed_size:]  # Keep only the up_proj gradients
                elif mod_name == 'self_attn.q_proj':
                    # q_proj is the first part of qkv_proj
                    activation = activation[:, :, :query_size]  # Keep only the q_proj part
                    grads = grads[:, :, :query_size]  # Keep only the q_proj gradients
                elif mod_name == 'self_attn.k_proj':
                    # k_proj is the second part of qkv_proj
                    activation = activation[:, :, query_size:query_size + key_size]  # Keep only the k_proj part
                    grads = grads[:, :, query_size:query_size + key_size]  # Keep only the k_proj gradients
                elif mod_name == 'self_attn.v_proj':
                    # v_proj is the third part of qkv_proj
                    activation = activation[:, :, query_size + key_size:]  # Keep only the v_proj part
                    grads = grads[:, :, query_size + key_size:]  # Keep only the v_proj gradients
                else:
                    raise ValueError(f"Unsupported module name: {mod_name} for model: {model.name_or_path}")
            else:
                raise ValueError(f"Unsupported model type: {model.name_or_path}")

            scores = activation * grads  # Element-wise multiplication to get scores (batch_size, seq_len, num_neurons)
            # ignore padding tokens
            padding_mask = attention_mask == 0
            if padding_mask.any():
                scores = scores.masked_fill(padding_mask.unsqueeze(-1), 0.0)  # Set scores for padding tokens to 0

            # aggregate scores with max across the sequence length
            max_aggr_scores, _ = torch.max(scores, dim=1)  # Shape: (batch_size, num_neurons

            # Normalize scores by the sum of probabilities for the correct token
            max_aggr_scores = prob_total * max_aggr_scores  # Shape: (batch_size, num_neurons)
            # for the sample from control dataset, set scores to max(0, score)
            max_aggr_scores[control] = torch.clamp(max_aggr_scores[control], min=0.0)  # Ensure scores are non-negative for control samples
            max_aggr_scores_cpu = max_aggr_scores.detach().cpu().tolist()  # Move to CPU for further processing

            # store scores for each sample and each neuron
            parts = module_name.split('.')
            layer_idx = int(parts[3])
            mod_name = '.'.join(parts[4:])

            # Loop over each sample to store scores
            for i in range(max_aggr_scores.size(0)):
                country = countries[i]
                for neuron_idx in range(max_aggr_scores.size(1)):
                    max_score = max_aggr_scores_cpu[i][neuron_idx]
                    max_neuron_scores[(mod_name, layer_idx, neuron_idx)][country] += max_score

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return max_neuron_scores, total_probabilities_per_country


def main():
    args = parse_args()
    logger = setup_logging()
    model_name = args.model_name
    dataset_names = args.dataset_names
    dataset_names.sort()  # Sort dataset names for consistent output
    logger.info(f"Calculating cultural neuron scores for model: {model_name} on dataset: {dataset_names}")

    # Load the model and tokenizer
    model, tokenizer = load_model(model_name)
    logger.info(f"Model {model_name} loaded successfully and moved to {model.device}.")

    max_neuron_scores = defaultdict(lambda: defaultdict(float))
    total_probabilities_per_country = defaultdict(float)

    # Calculate neuron scores
    dataloader = load_dataset_neuron_scores(
        dataset_names,
        tokenizer,
        batch_size=BATCH_SIZE,
        target_countries=None,   # Use all countries
        target_data='neuron'  # Only use neuron data
    )
    _max_neuron_scores, total_probs = calculate_scores(model, tokenizer, dataloader, logger)
    # Aggregate scores across all prompts
    for (module_name, layer_idx, neuron_idx), scores in _max_neuron_scores.items():
        for country, score in scores.items():
            max_neuron_scores[f'{module_name}_{layer_idx}_{neuron_idx}'][country] += score
    del _max_neuron_scores  # Free memory

    # Aggregate total probabilities for each country
    for country, prob in total_probs.items():
        total_probabilities_per_country[country] += prob

    # dataset IDs
    dataset_ids = defaultdict(list)
    for item in dataloader.dataset:
        if item['id'] not in dataset_ids[item['dataset_name']]:
            dataset_ids[item['dataset_name']].append(item['id'])

    # Save the neuron scores to a JSON file
    max_save_dict = {
        'neuron_scores': max_neuron_scores,
        'total_probabilities_per_country': total_probabilities_per_country,
        'dataset_ids': dataset_ids,
    }
    max_output_file = Path('../outputs') / args.model_name.split('/')[-1] / 'cultural_neuron' / f"{''.join(dataset_names)}_max_scores.json"
    max_output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(max_output_file, 'w') as f:
        json.dump(max_save_dict, f, indent=4)
    logger.info(f"Max neuron scores saved to {max_output_file}")

    # calculate neuron scores for countryrc dataset
    max_countryrc_neuron_scores = defaultdict(lambda: defaultdict(float))
    countryrc_total_probabilities_per_country = defaultdict(float)
    countryrc_dataloader = load_dataset_neuron_scores(
        dataset_names=['countryrc'],
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
        target_countries=TARGET_COUNTRIES,   # Use only target countries
        target_data='neuron'  # Only use neuron data
    )
    _max_neuron_scores_countryrc, total_probs_countryrc = calculate_scores(model, tokenizer, countryrc_dataloader, logger)
    # Aggregate scores across all prompts
    for (module_name, layer_idx, neuron_idx), scores in _max_neuron_scores_countryrc.items():
        for country, score in scores.items():
            max_countryrc_neuron_scores[f'{module_name}_{layer_idx}_{neuron_idx}'][country] += score
    # Aggregate total probabilities for each country
    for country, prob in total_probs_countryrc.items():
        countryrc_total_probabilities_per_country[country] += prob

    # Save the neuron scores for countryrc to a JSON file
    countryrc_dataset_ids = defaultdict(list)
    for item in countryrc_dataloader.dataset:
        if item['id'] not in countryrc_dataset_ids[item['dataset_name']]:
            countryrc_dataset_ids[item['dataset_name']].append(item['id'])
    max_save_dict_countryrc = {
        'neuron_scores': max_countryrc_neuron_scores,
        'total_probabilities_per_country': countryrc_total_probabilities_per_country,
        'dataset_ids': countryrc_dataset_ids,
    }
    max_output_file_countryrc = Path('../outputs') / args.model_name.split('/')[-1] / 'cultural_neuron' / f"countryrc_max_scores.json"
    max_output_file_countryrc.parent.mkdir(parents=True, exist_ok=True)
    with open(max_output_file_countryrc, 'w') as f:
        json.dump(max_save_dict_countryrc, f, indent=4)
    logger.info(f"Max neuron scores for countryrc saved to {max_output_file_countryrc}")


if __name__ == "__main__":
    set_seed(42)  # Set seed for reproducibility
    main()
