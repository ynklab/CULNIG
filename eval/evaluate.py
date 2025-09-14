import argparse
import logging
from pathlib import Path
import json
import sys

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

# add python path to import dataset and utils
sys.path.append(str(Path(__file__).resolve().parent.parent))
from dataset import load_dataset_neuron_scores
from utils import register_hook

BATCH_SIZE = 64
CRC_TARGET_COUNTRIES = [
    'China', 'Indonesia', 'Iran', 'Mexico', 'South Korea', 'Spain', 'UK', 'USA',
]   # countries must be specified for CountryRC
WVB_TARGET_COUNTRIES = [
    'China', 'Indonesia', 'Iran', 'Mexico', 'South Korea', 'UK', 'USA',
]   # Spain is not included in WVB


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LLMs on a specific dataset")
    parser.add_argument('--model_name', type=str, default='google/gemma-3-12b-it', help='Name of the model to use.')
    parser.add_argument('--neuron_file', type=str, default=None, help='Path to the neuron manipulation file. If provided, neurons will be manipulated.')
    parser.add_argument('--dataset_name', type=str, default='commonsenseqa', help='Name of the dataset to evaluate on. Default is commonsenseqa.')
    parser.add_argument('--operation', type=str, default='suppress', choices=['suppress', 'enhance'], help='Operation to perform on neurons: suppress or enhance.')
    return parser.parse_args()


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    logger = logging.getLogger(__name__)
    return logger


def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    return model, tokenizer


def evaluate_model(model, tokenizer, eval_dataset_name, logger, model_name, neuron_country=None, neuron_identification_dataset=None, neuron_data_ids=None, operation='suppress'):
    """Evaluate the model on the dataset."""

    results = []
    result_columns = ['instruction_id', 'id', 'predicted_label', 'gold_label', 'alignment_score']
    # alignment_score is only for WorldValuesBench

    # load dataset
    if eval_dataset_name == 'countryrc':
        target_countries = CRC_TARGET_COUNTRIES
    elif eval_dataset_name == 'worldvaluesbench':
        target_countries = WVB_TARGET_COUNTRIES
    else:
        target_countries = None
    dataloader = load_dataset_neuron_scores(
        dataset_names=[eval_dataset_name],
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
        target_countries=target_countries,
        target_data='all',
    )

    if eval_dataset_name == 'worldvaluesbench':
        wvb_data_root = Path(__file__).resolve().parent.parent / 'data' / 'WorldValuesBench'
        wvb_questions_path = wvb_data_root / 'question_metadata.json'
        with open(wvb_questions_path, 'r') as f:
            wvb_questions = json.load(f)

    total_batches = len(dataloader)
    cur_batch = 0
    model.eval()
    for batch in dataloader:
        cur_batch += 1
        if cur_batch % 200 == 0:
            logger.info(f"Processing batch {cur_batch}/{total_batches}...")
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        labels = batch['labels']
        ids = batch['ids']  # Get data indices
        instruction_ids = batch['instruction_ids']  # Get instruction IDs

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :]  # Get logits for the last token

        predictions = torch.argmax(logits, dim=-1)
        for i in range(len(predictions)):
            output = tokenizer.decode(predictions[i], skip_special_tokens=True)

            # alignment score
            if eval_dataset_name == 'worldvaluesbench':
                q_id = ids[i]
                min_option = wvb_questions[q_id]['answer_scale_min']
                max_option = wvb_questions[q_id]['answer_scale_max']
                gold_label = int(labels[i])
                if output.isdigit():
                    output = int(output)
                    if output < min_option or output > max_option:
                        alignment_score = 0.0
                    else:
                        alignment_score = (1 - abs(output - gold_label) / max(abs(min_option - gold_label), abs(max_option - gold_label))) * 100.0
                else:
                    alignment_score = 0.0
            else:
                alignment_score = None

            results.append({
                'instruction_id': instruction_ids[i],
                'id': str(ids[i]),
                'predicted_label': output,
                'gold_label': str(labels[i]),
                'alignment_score': alignment_score,
            })

    # save output files
    if neuron_country is None:
        output_dir = Path(__file__).parent.parent / 'outputs' / model_name.split('/')[-1] / eval_dataset_name
    else:
        output_dir = Path(__file__).parent.parent / 'outputs' / f'{model_name.split("/")[-1]}_{neuron_country}_{neuron_identification_dataset}' / eval_dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'results.csv' if operation == 'suppress' else output_dir / 'results_enhance.csv'
    results_df = pd.DataFrame(results, columns=result_columns)
    results_df['is_correct'] = results_df['predicted_label'] == results_df['gold_label']
    results_df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")

    # log accuracy
    accuracy = results_df['is_correct'].mean()
    logger.info(f'Overall Accuracy: {accuracy:.4f} ({(results_df["is_correct"]).sum()}/{len(results_df)})')
    if neuron_data_ids is not None:
        neuron_data_results_df = results_df[results_df['id'].isin(neuron_data_ids)]
        non_neuron_data_results_df = results_df[~results_df['id'].isin(neuron_data_ids)]
        neuron_accuracy = neuron_data_results_df['is_correct'].mean()
        non_neuron_accuracy = non_neuron_data_results_df['is_correct'].mean()
        logger.info(f'Neuron Data Accuracy: {neuron_accuracy:.4f} ({neuron_data_results_df['is_correct'].sum()}/{len(neuron_data_results_df)})')
        logger.info(f'Non-Neuron Data Accuracy: {non_neuron_accuracy:.4f} ({non_neuron_data_results_df['is_correct'].sum()}/{len(non_neuron_data_results_df)})')

    if eval_dataset_name == 'worldvaluesbench':
        # log alignment score
        alignment_score = results_df['alignment_score'].mean()
        logger.info(f'Overall Alignment Score: {alignment_score:.4f}')


def main():
    args = parse_args()
    logger = setup_logging()
    eval_dataset_name = args.dataset_name
    if args.model_name == 'google/gemma-3-27b-it':
        global BATCH_SIZE
        BATCH_SIZE = 32

    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model(args.model_name)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Model {args.model_name} loaded successfully.")
    neurons_to_manipulate = []
    if args.neuron_file:
        logger.info(f"Manipulating neurons using file: {args.neuron_file}")
        with open(args.neuron_file, 'r') as f:
            neuron_scores = json.load(f)
        top_neurons = neuron_scores['top_neurons']
        top_neurons.sort(key=lambda x: x['attribute_score'], reverse=True)  # Sort neurons by score in descending order
        neurons_to_manipulate = top_neurons
    neuron_country = Path(args.neuron_file).stem.split('_')[0] if args.neuron_file else None
    neuron_data_ids = neuron_scores['dataset_ids'][eval_dataset_name] if args.neuron_file and eval_dataset_name in neuron_scores['dataset_ids'] else None
    if args.neuron_file:
        neuron_file_stem = Path(args.neuron_file).stem
        if 'random' in neuron_file_stem:
            _, seed = neuron_file_stem.split('random')
            full_neuron_identification_dataset = f'random/seed{seed}'
        else:
            full_neuron_identification_dataset = '_'.join(Path(args.neuron_file).stem.split('_')[2:])
    else:
        full_neuron_identification_dataset = None

    operation = args.operation
    logger.info(f"Number of neurons to manipulate: {len(neurons_to_manipulate)}, Operation: {operation}, Neuron Country: {neuron_country} by {full_neuron_identification_dataset}")
    hooks = register_hook(model, neurons_to_manipulate, operation=operation) if neurons_to_manipulate else []

    logger.info(f"Evaluating model with {eval_dataset_name} dataset...")
    evaluate_model(model, tokenizer, eval_dataset_name, logger, args.model_name, neuron_country, full_neuron_identification_dataset, neuron_data_ids, operation)
    logger.info("Evaluation complete.")

    # Remove hooks after evaluation
    for hook in hooks:
        hook.remove()


if __name__ == "__main__":
    set_seed(42)  # Set seed for reproducibility
    main()
