import argparse
import logging
import json
from pathlib import Path
import random
import sys

from datasets import Dataset
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import register_hook


BATCH_SIZE = 32
COUNTRY_TO_NAME = {
    'USA': 'US', 'UK': 'UK', 'South Korea': 'South_Korea', 'Algeria': 'Algeria',
    'Indonesia': 'Indonesia', 'Spain': 'Spain', 'Iran': 'Iran', 'Mexico': 'Mexico',
    'Assam': 'Assam', 'Greece': 'Greece', 'Ethiopia': 'Ethiopia', 'Nigeria': 'Northern_Nigeria',
    'North Korea': 'North_Korea', 'West Java': 'West_Java', 'China': 'China', 'Azerbaijan': 'Azerbaijan',
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LLMs for BLEnD dataset")
    parser.add_argument('--model_name', type=str, default='google/gemma-3-12b-it', help='Name of the model to evaluate.')
    parser.add_argument('--neuron_file', type=str, default=None, help='Path to the neuron manipulation file. If provided, neurons will be manipulated.')
    parser.add_argument('--operation', type=str, default='suppress', choices=['suppress', 'enhance'], help='Operation to perform on neurons: suppress or enhance.')
    return parser.parse_args()


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    logger = logging.getLogger(__name__)
    return logger


def load_blend_sqa(tokenizer, batch_size=BATCH_SIZE):
    set_seed(42)  # For reproducibility

    # load BLEnD SQA
    data_dir = Path(__file__).resolve().parent.parent.parent / 'data' / 'BLEnD'
    data_countries = COUNTRY_TO_NAME.keys()

    categories = ['Food', 'Work life', 'Sport', 'Education', 'Family', 'Holidays/Celebration/Leisure']
    test_categories = categories[len(categories)//2:]  # Second half for evaluation
    metadata_path = Path(__file__).resolve().parent.parent.parent / 'data' / 'BLEnD' / 'US_questions.csv'
    metadata_df = pd.read_csv(metadata_path, encoding='utf-8')
    test_ids = metadata_df[metadata_df['Topic'].isin(test_categories)]['ID'].unique()

    all_data = []
    for country in data_countries:
        annotation_file = data_dir / 'annotations' / f'{COUNTRY_TO_NAME[country]}_data.json'
        prompt_file = data_dir / 'prompts' / f'{COUNTRY_TO_NAME[country]}_prompts.csv'
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
        prompts = pd.read_csv(prompt_file, dtype=str)
        prompts_candidates = prompts['Translation'].tolist()

        for anno_id in annotations:
            if anno_id not in test_ids:
                continue  # Skip non-test IDs
            annotation = annotations[anno_id]
            answers = []
            for anno in annotation['annotations']:
                if anno['count'] > 0:
                    answers.extend(anno['answers'])
            if not answers:
                continue  # Skip if no answers available
            answers = list(set(answers))  # Remove duplicates

            question = annotation['question']
            prompt = random.choice(prompts_candidates)   # randomly choose one prompt
            prompt_id = prompts[prompts['Translation'] == prompt]['id'].values[0]
            input_text = prompt.replace('{q}', question)
            try:
                input_text = tokenizer.apply_chat_template(
                    [{'role': 'user', 'content': input_text}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                add_special_tokens = False   # special tokens are already handled in the chat template
            except Exception as e:
                print(f"Error applying chat template: {e}")
                add_special_tokens = True
                pass
            # tokenize
            tokenized = tokenizer(input_text, return_tensors='pt', add_special_tokens=add_special_tokens)

            all_data.append({
                'input_ids': tokenized['input_ids'][0],
                'attention_mask': tokenized['attention_mask'][0],
                'answers': answers,
                'country': country,
                'id': anno_id,
                'instruction_id': prompt_id,
            })

    # create dataloader
    dataset = Dataset.from_list(all_data)
    def collate_fn(batch):
        input_ids = pad_sequence([torch.tensor(item['input_ids']) for item in batch], batch_first=True, padding_value=tokenizer.pad_token_id, padding_side='left')
        attention_mask = pad_sequence([torch.tensor(item['attention_mask']) for item in batch], batch_first=True, padding_value=0, padding_side='left')
        answers = [item['answers'] for item in batch]
        countries = [item['country'] for item in batch]
        ids = [item['id'] for item in batch]
        instruction_ids = [item['instruction_id'] for item in batch]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'answers': answers,
            'countries': countries,
            'ids': ids,
            'instruction_ids': instruction_ids,
        }
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return dataloader


def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    return model, tokenizer


def evaluate_model(model, tokenizer, logger, model_name, neuron_country=None, neuron_identification_dataset=None, operation='suppress'):
    """Evaluate the model on the NormAd dataset."""

    dataloader = load_blend_sqa(tokenizer, batch_size=BATCH_SIZE)

    results = []
    result_columns = ['instruction_id', 'id', 'country', 'prediction', 'answers']
    total_iter = len(dataloader)
    cur_iter = 0
    model.eval()
    for batch in dataloader:
        cur_iter += 1
        if cur_iter % 10 == 0:
            logger.info(f'Processing batch {cur_iter}/{total_iter}')

        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        answers = batch['answers']
        countries = batch['countries']
        ids = batch['ids']  # Get data indices
        instruction_ids = batch['instruction_ids']  # Get instruction IDs

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                max_new_tokens=512,
            )

        outputs = tokenizer.batch_decode(generated_ids[:, input_ids.shape[1]:], skip_special_tokens=True)
        for i in range(len(outputs)):
            output = outputs[i]
            results.append({
                'instruction_id': instruction_ids[i],
                'id': ids[i],
                'country': countries[i],
                'prediction': output,
                'answers': answers[i],
            })

    # save output files
    output_dir = Path(f'../outputs/{model_name.split("/")[-1]}/blend_sqa') if neuron_country is None else Path(f'../outputs/{model_name.split("/")[-1]}_{neuron_country}_{neuron_identification_dataset}/blend_sqa')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'results.csv' if operation == 'suppress' else output_dir / 'results_enhance.csv'
    results_df = pd.DataFrame(results, columns=result_columns)
    results_df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")


def main():
    args = parse_args()
    logger = setup_logging()

    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model(args.model_name)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')

    neurons_to_manipulate = []
    if args.neuron_file:
        logger.info(f"Manipulating neurons using file: {args.neuron_file}")
        with open(args.neuron_file, 'r') as f:
            neuron_scores = json.load(f)
        top_neurons = neuron_scores['top_neurons']
        top_neurons.sort(key=lambda x: x['attribute_score'], reverse=True)  # Sort neurons by score in descending order
        neurons_to_manipulate = top_neurons
    neuron_country = Path(args.neuron_file).stem.split('_')[0] if args.neuron_file else None
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
    logger.info(f"Number of neurons to manipulate: {len(neurons_to_manipulate)} - Operation: {operation}, Neuron Country: {neuron_country} by {full_neuron_identification_dataset}")
    hooks = register_hook(model, neurons_to_manipulate, operation=operation) if neurons_to_manipulate else []

    logger.info(f"Evaluating model {args.model_name}...")
    evaluate_model(model, tokenizer, logger, args.model_name, neuron_country, full_neuron_identification_dataset, operation)
    logger.info("Evaluation complete.")

    # Remove hooks after evaluation
    for hook in hooks:
        hook.remove()


if __name__ == "__main__":
    set_seed(42)  # Set seed for reproducibility
    main()
