import argparse
from pathlib import Path
import os
import logging
import psutil
import yaml
import sys

from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from transformers import set_seed
import wandb

# add python path to import dataset and utils
sys.path.append(str(Path(__file__).resolve().parent.parent))
from prompts import NEURON_PROMPTS

wandb.login(key=os.environ.get("WANDB_API_KEY", ""))


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LLMs on a specific dataset")
    parser.add_argument("--config", type=str, default=str(Path(__file__).resolve().parent / 'config.yaml'),
                        help="Path to the training configuration file")
    return parser.parse_args()


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    logger = logging.getLogger(__name__)
    return logger


def load_train_dataset(dataset_name, tokenizer, logger):
    # BLEnD dataset
    if dataset_name == 'blend':
        # instruction is already included in the dataset
        c2n = {
            'USA': 'US', 'UK': 'UK', 'South Korea': 'South_Korea', 'Algeria': 'Algeria',
            'Indonesia': 'Indonesia', 'Spain': 'Spain', 'Iran': 'Iran', 'Mexico': 'Mexico',
            'Assam': 'Assam', 'Greece': 'Greece', 'Ethiopia': 'Ethiopia', 'Nigeria': 'Northern_Nigeria',
            'North Korea': 'North_Korea', 'West Java': 'West_Java', 'China': 'China', 'Azerbaijan': 'Azerbaijan',
        }
        rev_c2n = {v: k for k, v in c2n.items()}  # Reverse mapping for country names
        dataset = load_dataset('nayeon212/BLEnD', 'multiple-choice-questions', split='test')

        # for each (country, ID), randomly select 5 samples
        country_id_mcqids = {}
        for item in dataset:
            country_id = (item['country'], item['ID'])
            if country_id not in country_id_mcqids:
                country_id_mcqids[country_id] = []
            if len(country_id_mcqids[country_id]) < 5:  # Limit to 5 samples per (country, ID)
                country_id_mcqids[country_id].append(item['MCQID'])
        valid_mcqids = []
        for mcqids in country_id_mcqids.values():
            valid_mcqids.extend(mcqids)  # Take up to 10 samples for each (country, ID)
        dataset = dataset.filter(lambda x: x['MCQID'] in valid_mcqids)  # Filter the dataset based on selected MCQIDs

        # split train/valid based on categories
        categories = ['Food', 'Work life', 'Sport', 'Education', 'Family', 'Holidays/Celebration/Leisure']
        train_categories = categories[:len(categories)//2]  # First half as neuron
        valid_categories = categories[len(categories)//2 : len(categories) // 2 + 1]  # one category for validation
        metadata_path = Path(__file__).resolve().parent.parent.parent / 'data' / 'BLEnD' / 'US_questions.csv'
        metadata_df = pd.read_csv(metadata_path, encoding='utf-8')
        train_neuron_ids = metadata_df[metadata_df['Topic'].isin(train_categories)]['ID'].unique()
        valid_neuron_ids = metadata_df[metadata_df['Topic'].isin(valid_categories)]['ID'].unique()
        train_dataset = dataset.filter(lambda x: x['ID'] in train_neuron_ids)
        valid_dataset = dataset.filter(lambda x: x['ID'] in valid_neuron_ids)

        def preprocess_function(examples):
            # No need to shuffle options as they are already balanced
            input_text = examples['prompt']
            try:
                input_text = tokenizer.apply_chat_template(
                    [{'role': 'user', 'content': input_text}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                input_text += '{"answer_choice":"'
                add_special_tokens = False   # special tokens are already handled in the chat template
            except Exception as e:
                print(f"Error applying chat template: {e}")
                add_special_tokens = True
                pass

            # in blend, the output is specified as JSON Format {"answer_choice":""}
            target_text = examples['answer_idx']

            # tokenize
            input_tokenized = tokenizer(input_text, return_tensors='pt', add_special_tokens=add_special_tokens)['input_ids'][0]
            target_tokenized = tokenizer(target_text, return_tensors='pt', add_special_tokens=False)['input_ids'][0]
            input_ids = torch.cat([input_tokenized, target_tokenized], dim=0)
            # attention mask
            attention_mask = torch.tensor([1] * len(input_tokenized) + [1] * len(target_tokenized))
            # labels
            labels = torch.tensor([-100] * len(input_tokenized) + target_tokenized.tolist())
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
            }
        train_dataset = train_dataset.map(preprocess_function, remove_columns=dataset.column_names, num_proc=1)  # Use single process for reproducibility
        valid_dataset = valid_dataset.map(preprocess_function, remove_columns=dataset.column_names, num_proc=1)  # Use single process for reproducibility
    elif dataset_name == 'mrpc':
        train_dataset = load_dataset('nyu-mll/glue', 'mrpc', split='train')  # Use validation split for MRPC
        valid_dataset = load_dataset('nyu-mll/glue', 'mrpc', split='validation')  # Use validation split for MRPC
        instruction = NEURON_PROMPTS['mrpc'][0]

        def preprocess_function(examples):
            input_text = instruction.format(sentence1=examples['sentence1'], sentence2=examples['sentence2'])
            target_text = str(examples['label'])
            try:
                input_text = tokenizer.apply_chat_template(
                    [{'role': 'user', 'content': input_text}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                add_special_tokens = False
            except Exception as e:
                print(f"Error applying chat template: {e}")
                add_special_tokens = True
                pass

            # tokenize
            input_tokenized = tokenizer(input_text, return_tensors='pt', add_special_tokens=add_special_tokens)['input_ids'][0]
            target_tokenized = tokenizer(target_text, return_tensors='pt', add_special_tokens=False)['input_ids'][0]
            input_ids = torch.cat([input_tokenized, target_tokenized], dim=0)
            # attention mask
            attention_mask = torch.tensor([1] * len(input_tokenized) + [1] * len(target_tokenized))
            # labels
            labels = torch.tensor([-100] * len(input_tokenized) + target_tokenized.tolist())
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
            }
        train_dataset = train_dataset.map(preprocess_function, remove_columns=train_dataset.column_names, num_proc=1)  # Use single process for reproducibility
        valid_dataset = valid_dataset.map(preprocess_function, remove_columns=valid_dataset.column_names, num_proc=1)  # Use single process for reproducibility
    elif dataset_name == 'qnli':
        train_dataset = load_dataset('nyu-mll/glue', 'qnli', split='train')
        # randomly select 10,000 samples
        train_dataset = train_dataset.shuffle(seed=42).select([i for i in list(range(10000))])
        instruction = NEURON_PROMPTS['qnli'][0]

        def preprocess_function(examples):
            input_text = instruction.format(question=examples['question'], sentence=examples['sentence'])
            target_text = str(examples['label'])
            try:
                input_text = tokenizer.apply_chat_template(
                    [{'role': 'user', 'content': input_text}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                add_special_tokens = False
            except Exception as e:
                print(f"Error applying chat template: {e}")
                add_special_tokens = True
                pass

            # tokenize
            input_tokenized = tokenizer(input_text, return_tensors='pt', add_special_tokens=add_special_tokens)['input_ids'][0]
            target_tokenized = tokenizer(target_text, return_tensors='pt', add_special_tokens=False)['input_ids'][0]
            input_ids = torch.cat([input_tokenized, target_tokenized], dim=0)
            # attention mask
            attention_mask = torch.tensor([1] * len(input_tokenized) + [1] * len(target_tokenized))
            # labels
            labels = torch.tensor([-100] * len(input_tokenized) + target_tokenized.tolist())
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
            }
        train_dataset = train_dataset.map(preprocess_function, remove_columns=train_dataset.column_names, num_proc=1)  # Use single process for reproducibility
        valid_dataset = None
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Create a DataLoader
    def collator(batch):
        input_ids = pad_sequence([torch.tensor(item['input_ids']) for item in batch], batch_first=True, padding_value=tokenizer.pad_token_id, padding_side='left')
        attention_mask = pad_sequence([torch.tensor(item['attention_mask']) for item in batch], batch_first=True, padding_value=0, padding_side='left')
        labels = pad_sequence([torch.tensor(item['labels']) for item in batch], batch_first=True, padding_value=-100, padding_side='left')
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
    return {
        'train': train_dataset,
        'valid': valid_dataset,
        'collator': collator,
    }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # logits: [batch_size, seq_len, vocab_size]
    # labels: [batch_size, seq_len]

    # we only compute accuracy for the last token
    preds = []
    refs = []

    for logit, label in zip(logits, labels):
        valid_positions = label != -100
        if valid_positions.any():
            last_idx = torch.where(valid_positions)[0][-1]
            pred_token_id = np.argmax(logit[last_idx])
            true_token_id = label[last_idx]
            preds.append(pred_token_id)
            refs.append(true_token_id)

    acc = accuracy_score(refs, preds)
    return {"accuracy": acc}


def load_model(model_name, logger):
    logger.info(f"model name: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation='eager' if 'gemma-3' in model_name else 'sdpa',
    )
    model = model.to('cuda')
    logger.info(f'successfully loaded model and tokenizer.')
    return model, tokenizer


def init_lora(model, lora_config, target_modules=None, target_layers=None):
    lora_config = LoraConfig(
        r=lora_config['r'],
        lora_alpha=lora_config['lora_alpha'],
        lora_dropout=lora_config['lora_dropout'],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
        layers_to_transform=[i for i in range(target_layers[0], target_layers[1] + 1)] if target_layers else None,
    )
    # model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def main():
    args = parse_args()
    logger = setup_logging()

    # load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    lora_config = config['train']['lora']

    # set seed
    set_seed(config.get('seed', 42))

    # load model and tokenizer
    model, tokenizer = load_model(config['model']['model_name_or_path'], logger)
    logger.info(f'GPU memory usage after model loading: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB')
    logger.info(f'CPU memory usage after model loading: {psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2):.2f} MB')

    # load dataset
    dataset = load_train_dataset(config['dataset_name'], tokenizer, logger)
    train_dataset = dataset['train']
    valid_dataset = dataset['valid']
    collator = dataset['collator']
    logger.info(f"successfully loaded dataset.")
    logger.info(f'Loaded dataset size: {len(train_dataset)} training samples')

    # initialize LoRA or set target modules/layers to trainable
    if config['train']['lora']['enabled']:
        model = init_lora(model, lora_config, config['train'].get('target_modules', None), config['train'].get('target_layers', None))
    else:
        logger.info("LoRA is not enabled. Proceeding without LoRA.")
        if config['train'].get('target_modules', None):
            for name, param in model.named_parameters():
                if 'language_model' in name and any(module in name for module in config['train']['target_modules']):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            logger.info("Target modules set to trainable.")
        else:
            logger.info("No target modules specified. All parameters are trainable.")
            for name, param in model.named_parameters():
                if 'language_model' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        if config['train'].get('target_layers', None):
            start_layer_idx = config['train']['target_layers'][0]
            end_layer_idx = config['train']['target_layers'][1]
            for name, param in model.named_parameters():
                if 'language_model' in name and 'layers' in name:
                    layer_idx = int(name.split('.')[3])
                    if layer_idx < start_layer_idx or layer_idx > end_layer_idx:
                        param.requires_grad = False
            logger.info(f"Layers {start_layer_idx} to {end_layer_idx} set to trainable.")

        all_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {all_params}, Trainable parameters: {trainable_params} ({100 * trainable_params / all_params:.2f}%)")

    training_args = TrainingArguments(
        output_dir=config['train']['output_dir'],
        run_name=config['train']['run_name'],
        bf16=True,
        per_device_train_batch_size=config['train']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['train']['per_device_eval_batch_size'],
        gradient_accumulation_steps=config['train']['gradient_accumulation_steps'],
        num_train_epochs=config['train']['num_train_epochs'],
        learning_rate=config['train']['learning_rate'],
        warmup_ratio=config['train']['warmup_ratio'],
        logging_steps=config['train']['logging_steps'],
        save_strategy=config['train']['save_strategy'],
        save_steps=config['train'].get('save_steps', None),
        eval_strategy=config['train']['eval_strategy'],
        eval_steps=config['train'].get('eval_steps', None),
        eval_accumulation_steps=config['train'].get('eval_accumulation_steps', 1),
        lr_scheduler_type=config['train']['lr_scheduler_type'],
        optim=config['train']['optim'],
        save_only_model=True,
        report_to="wandb",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    # start training
    trainer.train()


if __name__ == "__main__":
    main()
