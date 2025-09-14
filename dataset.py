from collections import defaultdict
import random
import hashlib
from pathlib import Path
import json
import itertools

from datasets import load_dataset, concatenate_datasets, Dataset
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.distributed import DistributedSampler

from prompts import NEURON_PROMPTS, NEURON_SYSTEM_PROMPTS

DATA_DIR = Path(__file__).resolve().parent / 'data'
COUNTRY_TO_NAME = {
    'normad': {
        'USA': 'united_states_of_america', 'China': 'china', 'Germany': 'germany', 'Japan': 'japan',
        'Mexico': 'mexico', 'India': 'india', 'Indonesia': 'indonesia', 'Russia': 'russia', 'Brazil': 'brazil',
        'Iran': 'iran', 'Zimbabwe': 'zimbabwe', 'Spain': 'spain', 'UK': 'united_kingdom', 'South Korea': 'south_korea',
    },
    'culturalbench': {
        'Brazil': 'Brazil', 'China': 'China', 'Germany': 'Germany', 'India': 'India', 'Indonesia': 'Indonesia',
        'Iran': 'Iran', 'Japan': 'Japan', 'Mexico': 'Mexico', 'Russia': 'Russia', 'South Korea': 'South Korea',
        'Spain': 'Spain', 'UK': 'United Kingdom', 'USA': 'United States', 'Zimbabwe': 'Zimbabwe',
    },
    'blend': {
        'USA': 'US', 'UK': 'UK', 'South Korea': 'South_Korea', 'Algeria': 'Algeria',
        'Indonesia': 'Indonesia', 'Spain': 'Spain', 'Iran': 'Iran', 'Mexico': 'Mexico',
        'Assam': 'Assam', 'Greece': 'Greece', 'Ethiopia': 'Ethiopia', 'Nigeria': 'Northern_Nigeria',
        'North Korea': 'North_Korea', 'West Java': 'West_Java', 'China': 'China', 'Azerbaijan': 'Azerbaijan',
    },
    'worldvaluesbench': {
        'China': 'China', 'Mexico': 'Mexico', 'Indonesia': 'Indonesia', 'Iran': 'Iran', 'South Korea': 'South Korea',
        'UK': 'Great Britain', 'USA': 'United States',
    },
}


def load_dataset_neuron_scores(dataset_names, tokenizer, batch_size, target_countries=None, target_data='all'):
    """
    Load the dataset for calculating neuron scores.
    
    Args:
        dataset_names (List[str]): The list of the names of the dataset to load.
        tokenizer (Tokenizer): The tokenizer to use for encoding the dataset.
        batch_size (int): The batch size for processing the dataset.
        target_countries (list, optional): List of target countries. If None, all countries are used.
        target_data (str, optional): Specifies the target data type.
            choices are 'all', 'neuron', 'non_neuron'.

    Returns:
        Dataloader: A DataLoader object containing the dataset. Following the structure:
            {
                'input_text': str,
                'input_ids': torch.Tensor,
                'attention_mask': torch.Tensor,
                'labels': str,  # The label for the sample, which is the index of the selected option
                'country': str,
                'id': int,  # Unique identifier for the sample,
                'instruction_id': int,  # Instruction index for the sample,
                'dataset_name': str,  # Name of the dataset
                'options': List[str],  # List of options for the question
            }
    """
    assert target_data in ['all', 'neuron', 'non_neuron'], "target_data must be one of 'all', 'neuron', or 'non_neuron'."

    # Set random seed for reproducibility at the function level
    random.seed(42)
    datasets = []

    # NormAd dataset
    if 'normad' in dataset_names:
        instructoins = NEURON_PROMPTS['normad']
        c2n = COUNTRY_TO_NAME['normad']
        rev_c2n = {v: k for k, v in c2n.items()}  # Reverse mapping for country names
        dataset = load_dataset('akhilayerukola/NormAd', split='train')   # only train
        if target_countries is not None:
            normad_target_countries = [c2n[country] for country in target_countries]  # Convert country names to NormAd format
            dataset = dataset.filter(lambda x: x['Country'] in normad_target_countries)  # Filter for target countries
        else:
            # all countries are used
            normad_target_countries = np.unique(dataset['Country']).tolist()  # Get all unique countries in the dataset

        if target_data != 'all':
            # options are yes, no, and neutral
            # For each country and each label, use half of the samples as neuron data and the other half as non-neuron data
            IDs = []
            for country in normad_target_countries:
                country_data = dataset.filter(lambda x: x['Country'] == country)
                for label in ['yes', 'no', 'neutral']:
                    label_data = country_data.filter(lambda x: x['Gold Label'] == label)
                    n_samples = len(label_data)
                    n_neuron = n_samples // 2  # Use half of the samples as neuron data
                    neuron_samples = label_data.select(range(n_neuron))
                    non_neuron_samples = label_data.select(range(n_neuron, n_samples))
                    if target_data == 'neuron':
                        IDs.extend(neuron_samples['ID'])
                    elif target_data == 'non_neuron':
                        IDs.extend(non_neuron_samples['ID'])
            dataset = dataset.filter(lambda x: x['ID'] in IDs)  # Filter the dataset based on selected IDs
        else:
            # Use all data
            pass

        for inst_idx, instruction in enumerate(instructoins):
            def preprocess_function(examples):
                # Use stable hash for reproducible shuffling
                hash_input = f"{examples['ID']}_{instruction}"
                problem_seed = int(hashlib.md5(hash_input.encode()).hexdigest()[:8], 16) % (2**32)
                rng = random.Random(problem_seed)  # Create a new random generator with problem-specific seed
                
                # randomly select option indices using problem-specific RNG
                options = [1, 2, 3]
                rng.shuffle(options)
                option_labels = {options[0]: 'yes', options[1]: 'no', options[2]: 'neutral'}
                option_labels = sorted(option_labels.items(), key=lambda x: x[0])  # Sort by option index
                option_str = f'{option_labels[0][0]}: {option_labels[0][1]}, {option_labels[1][0]}: {option_labels[1][1]}, {option_labels[2][0]}: {option_labels[2][1]}'
                if examples['Gold Label'] == 'yes':
                    label = options[0]
                elif examples['Gold Label'] == 'no':
                    label = options[1]
                elif examples['Gold Label'] == 'neutral':
                    label = options[2]
                else:
                    raise ValueError(f"Unknown label: {examples['Gold Label']}")

                input_text = instruction.format(country=examples['Country'], story=examples['Story'], options=option_str)
                try:
                    input_text = tokenizer.apply_chat_template(
                        [{'role': 'user', 'content': input_text}],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    add_special_tokens = False   # special tokens are already handled in the chat template
                except Exception as e:
                    add_special_tokens = True
                    pass
                tokenized = tokenizer(input_text, return_tensors='pt', add_special_tokens=add_special_tokens)

                return {
                    'input_text': input_text,
                    'input_ids': tokenized['input_ids'][0],
                    'attention_mask': tokenized['attention_mask'][0],
                    'label': str(label),
                    'country': rev_c2n[examples['Country']] if examples['Country'] in rev_c2n else examples['Country'],  # Use reverse mapping for country names
                    'id': str(examples['ID']),
                    'instruction_id': inst_idx,  # Add instruction index for reproducibility
                    'dataset_name': 'normad',  # Add dataset name for identification
                    'options': [str(opt) for opt in options],  # Store the shuffled options
                }
            dataset_processed = dataset.map(preprocess_function, remove_columns=dataset.column_names, num_proc=1)  # Use single process for reproducibility
            datasets.append(dataset_processed)

    # CulturalBench dataset
    if 'culturalbench' in dataset_names:
        instruction = NEURON_PROMPTS['culturalbench']
        c2n = COUNTRY_TO_NAME['culturalbench']
        rev_c2n = {v: k for k, v in c2n.items()}  # Reverse mapping for country names
        dataset = load_dataset('kellycyy/CulturalBench', "CulturalBench-Easy", split='test')   # testしかない
        if target_countries is not None:
            culturalbench_target_countries = [c2n[country] for country in target_countries]
            dataset = dataset.filter(lambda x: x['country'] in culturalbench_target_countries)
        else:
            # all countries are used
            culturalbench_target_countries = np.unique(dataset['country']).tolist()  # Get all unique countries in the dataset

        if target_data != 'all':
            # For each country, use half of the samples as neuron data and the other half as non-neuron data
            IDs = []
            for country in culturalbench_target_countries:
                country_data = dataset.filter(lambda x: x['country'] == country)
                n_samples = len(country_data)
                n_neuron = n_samples // 2  # Use half of the samples as neuron data
                neuron_samples = country_data.select(range(n_neuron))
                non_neuron_samples = country_data.select(range(n_neuron, n_samples))
                if target_data == 'neuron':
                    IDs.extend(neuron_samples['data_idx'])
                elif target_data == 'non_neuron':
                    IDs.extend(non_neuron_samples['data_idx'])
            dataset = dataset.filter(lambda x: x['data_idx'] in IDs)  # Filter the dataset based on selected IDs
        else:
            # Use all data
            pass

        for inst_idx, instruction in enumerate(instruction):
            def preprocess_function(examples):
                # Use stable hash for reproducible shuffling
                hash_input = f"{examples['data_idx']}_{instruction}"
                problem_seed = int(hashlib.md5(hash_input.encode()).hexdigest()[:8], 16) % (2**32)
                rng = random.Random(problem_seed)  # Create a new random generator with problem-specific seed
                
                option_ans = [examples['prompt_option_a'], examples['prompt_option_b'], examples['prompt_option_c'], examples['prompt_option_d']]
                rng.shuffle(option_ans)  # Shuffle options using problem-specific RNG to avoid bias
                if '1. ' in instruction:
                    options = ['1', '2', '3', '4']
                else:
                    options = ['A', 'B', 'C', 'D']

                label_idx = option_ans.index(examples[f'prompt_option_{examples['answer'].lower()}'])
                label = options[label_idx]  # Get the label based on the shuffled options

                input_text = instruction.format(
                    question=examples['prompt_question'],
                    option_a=option_ans[0],
                    option_b=option_ans[1],
                    option_c=option_ans[2],
                    option_d=option_ans[3],
                )
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
                return {
                    'input_text': input_text,
                    'input_ids': tokenized['input_ids'][0],
                    'attention_mask': tokenized['attention_mask'][0],
                    'label': label,
                    'country': rev_c2n[examples['country']] if examples['country'] in rev_c2n else examples['country'],  # Use reverse mapping for country names
                    'id': str(examples['data_idx']),
                    'instruction_id': inst_idx,  # CulturalBench has only one instruction
                    'dataset_name': 'culturalbench',  # Add dataset name for identification
                    'options': options,  # Store the shuffled options
                }
            dataset_processed = dataset.map(preprocess_function, remove_columns=dataset.column_names, num_proc=1)  # Use single process for reproducibility
            datasets.append(dataset_processed)

    # BLEnD dataset
    if 'blend' in dataset_names:
        # instruction is already included in the dataset
        c2n = COUNTRY_TO_NAME['blend']
        rev_c2n = {v: k for k, v in c2n.items()}  # Reverse mapping for country names
        dataset = load_dataset('nayeon212/BLEnD', 'multiple-choice-questions', split='test')   # testしかない
        if target_countries is not None:
            blend_target_countries = [c2n[country] for country in target_countries]
            dataset = dataset.filter(lambda x: x['country'] in blend_target_countries)
        else:
            # all countries are used
            blend_target_countries = np.unique(dataset['country']).tolist()

        # In BLEnD, the content of the questions varies between combinations of (ID, country),
        # with up to several hundred instances differing only in options within the combination.
        # Therefore, we adopt a maximum of 5 samples for each (ID, country).
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

        if target_data != 'all':
            # For each question category, half are neuron and half are non-neuron
            categories = ['Food', 'Work life', 'Sport', 'Education', 'Family', 'Holidays/Celebration/Leisure']
            neuron_categories = categories[:len(categories)//2]  # First half as neuron
            non_neuron_categories = categories[len(categories)//2:]  # Second half as non-neuron
            metadata_path = DATA_DIR / 'BLEnD' / 'US_questions.csv'
            metadata_df = pd.read_csv(metadata_path, encoding='utf-8')
            neuron_ids = metadata_df[metadata_df['Topic'].isin(neuron_categories)]['ID'].unique()
            non_neuron_ids = metadata_df[metadata_df['Topic'].isin(non_neuron_categories)]['ID'].unique()
            if target_data == 'neuron':
                dataset = dataset.filter(lambda x: x['ID'] in neuron_ids)
            elif target_data == 'non_neuron':
                dataset = dataset.filter(lambda x: x['ID'] in non_neuron_ids)
        else:
            # Use all data
            pass

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
                # in blend, the output is specified as JSON Format {"answer_choice":""}
                input_text += '{"answer_choice":"'
                add_special_tokens = False   # special tokens are already handled in the chat template
            except Exception as e:
                print(f"Error applying chat template: {e}")
                add_special_tokens = True
                pass

            # tokenize
            tokenized = tokenizer(input_text, return_tensors='pt', add_special_tokens=add_special_tokens)
            return {
                'input_text': input_text,
                'input_ids': tokenized['input_ids'][0],
                'attention_mask': tokenized['attention_mask'][0],
                'label': examples['answer_idx'],
                'country': rev_c2n[examples['country']] if examples['country'] in rev_c2n else examples['country'],  # Use reverse mapping for country names
                'id': examples['MCQID'],
                'instruction_id': 0,  # BLEnD has only one instruction
                'dataset_name': 'blend',  # Add dataset name for identification
                'options': ['A', 'B', 'C', 'D'],  # Options are fixed as A, B, C, D
            }
        dataset_processed = dataset.map(preprocess_function, remove_columns=dataset.column_names, num_proc=1)  # Use single process for reproducibility
        datasets.append(dataset_processed)

    # BLEnD Control dataset
    if 'blendcontrol' in dataset_names:
        # instruction is already included in the dataset
        c2n = COUNTRY_TO_NAME['blend']
        rev_c2n = {v: k for k, v in c2n.items()}  # Reverse mapping for country names
        dataset = load_dataset('nayeon212/BLEnD', 'multiple-choice-questions', split='test')   # testしかない
        if target_countries is not None:
            blend_target_countries = [c2n[country] for country in target_countries]
            dataset = dataset.filter(lambda x: x['country'] in blend_target_countries)
        else:
            # all countries are used
            blend_target_countries = np.unique(dataset['country']).tolist()

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

        if target_data != 'all':
            # For each question category, half are neuron and half are non-neuron
            categories = ['Food', 'Work life', 'Sport', 'Education', 'Family', 'Holidays/Celebration/Leisure']
            neuron_categories = categories[:len(categories)//2]  # First half as neuron
            non_neuron_categories = categories[len(categories)//2:]  # Second half as non-neuron
            metadata_path = DATA_DIR / 'BLEnD' / 'US_questions.csv'
            metadata_df = pd.read_csv(metadata_path, encoding='utf-8')
            neuron_ids = metadata_df[metadata_df['Topic'].isin(neuron_categories)]['ID'].unique()
            non_neuron_ids = metadata_df[metadata_df['Topic'].isin(non_neuron_categories)]['ID'].unique()
            if target_data == 'neuron':
                dataset = dataset.filter(lambda x: x['ID'] in neuron_ids)
            elif target_data == 'non_neuron':
                dataset = dataset.filter(lambda x: x['ID'] in non_neuron_ids)
        else:
            # Use all data
            pass

        control_input_text = "Without any explanation, choose only one from the given alphabet choices(e.g., A, B, C). Provide as JSON format: {{\"answer_choice\":\"\"}}\n\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}\n\nAnswer:"
        def preprocess_function(examples):
            # No need to shuffle options as they are already balanced
            choices = json.loads(examples['choices'])
            input_text = control_input_text.format(
                option_a=choices['A'],
                option_b=choices['B'],
                option_c=choices['C'],
                option_d=choices['D'],
            )
            try:
                input_text = tokenizer.apply_chat_template(
                    [{'role': 'user', 'content': input_text}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                # in blend, the output is specified as JSON Format {"answer_choice":""}
                input_text += '{"answer_choice":"'
                add_special_tokens = False   # special tokens are already handled in the chat template
            except Exception as e:
                print(f"Error applying chat template: {e}")
                add_special_tokens = True
                pass

            # tokenize
            tokenized = tokenizer(input_text, return_tensors='pt', add_special_tokens=add_special_tokens)
            return {
                'input_text': input_text,
                'input_ids': tokenized['input_ids'][0],
                'attention_mask': tokenized['attention_mask'][0],
                'label': examples['answer_idx'],
                'country': rev_c2n[examples['country']] if examples['country'] in rev_c2n else examples['country'],  # Use reverse mapping for country names
                'id': examples['MCQID'],
                'instruction_id': 0,  # BLEnD has only one instruction
                'dataset_name': 'blendcontrol',  # Add dataset name for identification
                'options': ['A', 'B', 'C', 'D'],  # Options are fixed as A, B, C, D
            }
        dataset_processed = dataset.map(preprocess_function, remove_columns=dataset.column_names, num_proc=1)  # Use single process for reproducibility
        datasets.append(dataset_processed)

    # WorldValuesBench
    if 'worldvaluesbench' in dataset_names:
        assert target_countries is not None, "target_countries must be specified for worldvaluesbench dataset."

        wvb_data_root = DATA_DIR / 'WorldValuesBench'
        instructions = NEURON_PROMPTS['worldvaluesbench']
        system_prompts = NEURON_SYSTEM_PROMPTS['worldvaluesbench']
        c2n = COUNTRY_TO_NAME['worldvaluesbench']
        rev_c2n = {v: k for k, v in c2n.items()}
        worldvaluesbench_target_countries = [c2n[country] for country in target_countries]

        # sort the questions by the distance from the mean of the distribution
        def calculate_distance(dist_all, dist_country):
            """Calculate the KL Divergence between the country distribution and the overall distribution."""
            total_all = sum(dist_all.values())
            total_country = sum(dist_country.values())
            prob_all = {k: v / total_all for k, v in dist_all.items()}
            prob_country = {k: v / total_country for k, v in dist_country.items()}
            # calculate KL divergence KL(dist_country || dist_all)
            kl_div = 0.0
            for k in set(prob_all.keys()).union(prob_country.keys()):
                p_all = prob_all.get(k, 0.0)
                p_country = prob_country.get(k, 0.0)
                if p_country > 0 and p_all > 0:
                    kl_div += p_country * np.log(p_country / p_all)
            return kl_div.item()

        wvb_questions_path = wvb_data_root / 'question_metadata.json'
        with open(wvb_questions_path, 'r', encoding='utf-8') as f:
            wvb_questions = json.load(f)
        split = 'full'
        wvb_data_dir = wvb_data_root / split
        wvb_demographic_path = wvb_data_dir / f'{split}_demographic_qa.tsv'
        wvb_demographic_df = pd.read_csv(wvb_demographic_path, sep='\t', encoding='utf-8')
        wvb_value_path = wvb_data_dir / f'{split}_value_qa.tsv'
        wvb_value_df = pd.read_csv(wvb_value_path, sep='\t', encoding='utf-8')
        wvb_dist_all = defaultdict(lambda: defaultdict(int))  # wvb_dist[Q{i}][{option}] = count
        for _, row in wvb_value_df.iterrows():
            for i in range(1, 260):
                # Q1 ~ Q259
                question_key = f'Q{i}'
                if question_key not in wvb_questions:
                    continue
                if pd.isna(row[question_key]) or row[question_key] == '':
                    continue
                answer = int(row[question_key])
                wvb_dist_all[question_key][answer] += 1
        # filter out questions with less than 10 responses
        wvb_dist_all = {k: v for k, v in wvb_dist_all.items() if sum(v.values()) >= 10}

        data = []  # List to hold the processed data
        for country in worldvaluesbench_target_countries:
            country_wvb_demographic_df = wvb_demographic_df[wvb_demographic_df['B_COUNTRY'] == country]

            # select only the rows that have the same D_INTERVIEW as in the demographic data
            country_interviews = country_wvb_demographic_df['D_INTERVIEW'].unique()
            country_wvb_value_df = wvb_value_df[wvb_value_df['D_INTERVIEW'].isin(country_interviews)]

            # aggregate the data for each question
            wvb_dist_country = defaultdict(lambda: defaultdict(int))   # wvb_dist[Q{i}][{option}] = count
            for _, row in country_wvb_value_df.iterrows():
                for i in range(1, 260):
                    # Q1 ~ Q259
                    question_key = f'Q{i}'
                    if question_key not in wvb_questions:
                        continue
                    if pd.isna(row[question_key]) or row[question_key] == '':
                        continue
                    answer = int(row[question_key])
                    wvb_dist_country[question_key][answer] += 1
            # filter out questions with less than 10 responses
            wvb_dist_country = {k: v for k, v in wvb_dist_country.items() if sum(v.values()) >= 10}

            wvb_dist_country = sorted(
                wvb_dist_country.items(),
                key=lambda x: calculate_distance(wvb_dist_all[x[0]], x[1]),
                reverse=True,
            )

            # for neuron data, select the top 40 questions based on distance
            if target_data == 'all':
                wvb_dist = wvb_dist_country
            elif target_data == 'neuron':
                wvb_dist = wvb_dist_country[:40]  # Select top 40 questions for neuron
            elif target_data == 'non_neuron':
                wvb_dist = wvb_dist_country[40:]  # Select remaining questions for non-neuron

            for question, dist in wvb_dist:
                question_sentence = wvb_questions[question]['question']
                min_option = wvb_questions[question]['answer_scale_min']
                max_option = wvb_questions[question]['answer_scale_max']
                data.append({
                    'Q_ID': question,
                    'question': question_sentence,
                    'options': list(range(min_option, max_option + 1)),
                    'distribution': dist,
                    'country': country,
                })

        # Convert the data to transformers Dataset format
        dataset_dict = {
            'input_text': [],
            'input_ids': [],
            'attention_mask': [],
            'label': [],
            'country': [],
            'id': [],
            'instruction_id': [],
            'dataset_name': [],
            'options': [],
        }
        for system_prompt_idx, system_prompt in enumerate(system_prompts):
            for inst_idx, instruction in enumerate(instructions):
                for item in data:
                    country = item['country']
                    question = item['question']
                    options = item['options']
                    min_option = min(options)
                    max_option = max(options)
                    system_prompt_formatted = system_prompt.format(country=country)
                    prompt = instruction.format(
                        question=question,
                        min_option=min_option,
                        max_option=max_option,
                    )
                    try:
                        prompt = tokenizer.apply_chat_template(
                            [
                                {'role': 'system', 'content': system_prompt_formatted},
                                {'role': 'user', 'content': prompt},
                            ],
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=False,
                        )
                        add_special_tokens = False   # special tokens are already handled in the chat template
                    except Exception as e:
                        print(f"Error applying chat template: {e}")
                        add_special_tokens = True
                        pass
                    tokenized = tokenizer(prompt, return_tensors='pt', add_special_tokens=add_special_tokens)
                    # select the most frequent option as the label
                    majority_option = max(item['distribution'], key=item['distribution'].get)
                    dataset_dict['input_text'].append(prompt)
                    dataset_dict['input_ids'].append(tokenized['input_ids'][0])
                    dataset_dict['attention_mask'].append(tokenized['attention_mask'][0])
                    dataset_dict['label'].append(str(majority_option))  # Convert to string for consistency
                    dataset_dict['country'].append(rev_c2n[country] if country in rev_c2n else country)
                    dataset_dict['id'].append(item['Q_ID'])
                    dataset_dict['instruction_id'].append(system_prompt_idx * len(instructions) + inst_idx)  # Unique instruction index for reproducibility
                    dataset_dict['dataset_name'].append('worldvaluesbench')  # Add dataset name for identification
                    dataset_dict['options'].append([str(opt) for opt in options])  # Store the options as strings

        # Create a Dataset from the dictionary
        dataset_processed = Dataset.from_dict(dataset_dict)
        datasets.append(dataset_processed)

    # CountryRC dataset
    if 'countryrc' in dataset_names:
        assert target_countries is not None, "target_countries must be specified for countryrc dataset."
        assert len(target_countries) > 0, "target_countries must not be empty."
        instructions = NEURON_PROMPTS['countryrc']
        crc_dir = DATA_DIR / 'CountryRC'
        crc_data_path = crc_dir / 'data.json'
        with open(crc_data_path, 'r', encoding='utf-8') as f:
            crc_data = json.load(f)

        if target_data != 'all':
            # halve the data into neuron and non-neuron based on the order in the file
            data_num = len(crc_data)
            neuron_data_num = data_num // 2
            neuron_data = crc_data[:neuron_data_num]
            non_neuron_data = crc_data[neuron_data_num:]
            if target_data == 'neuron':
                crc_data = neuron_data
            elif target_data == 'non_neuron':
                crc_data = non_neuron_data
        else:
            # Use all data
            pass

        dataset_dict = {
            'input_text': [],
            'input_ids': [],
            'attention_mask': [],
            'label': [],
            'country': [],
            'id': [],
            'instruction_id': [],
            'dataset_name': [],
            'options': [],
        }
        for country in target_countries:
            for inst_idx, instruction in enumerate(instructions):
                for item in crc_data:
                    context = item['context']
                    question = item['question']

                    # seed
                    hash_input = f"{context}_{question}_{instruction}_{country}"
                    problem_seed = int(hashlib.md5(hash_input.encode()).hexdigest()[:8], 16) % (2**32)
                    rng = random.Random(problem_seed)  # Create a new random generator with problem-specific seed

                    # make options
                    country_pool = [c for c in COUNTRY_TO_NAME['blend'].keys() if c != country]
                    # select 3 random countries from the pool
                    options = rng.sample(country_pool, 3)
                    options.append(country)  # Add the target country as the last option
                    rng.shuffle(options)  # Shuffle options using problem-specific RNG to avoid bias
                    if '1. ' in instruction:
                        options_str = ['1', '2', '3', '4']
                    else:
                        options_str = ['A', 'B', 'C', 'D']
                    label_idx = options.index(country)  # Get the index of the target country
                    label = options_str[label_idx]  # Get the label based on the shuffled options

                    if 'country_dummy' in context:
                        # select dummy country
                        dummy_options = [c for c in options if c != country]
                        country_dummy = rng.choice(dummy_options)
                        context = context.format(country=country, country_dummy=country_dummy)
                    else:
                        context = context.format(country=country)

                    input_text = instruction.format(
                        passage=context,
                        question=question,
                        option_a=options[0],
                        option_b=options[1],
                        option_c=options[2],
                        option_d=options[3],
                    )
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
                    dataset_dict['input_text'].append(input_text)
                    dataset_dict['input_ids'].append(tokenized['input_ids'][0])
                    dataset_dict['attention_mask'].append(tokenized['attention_mask'][0])
                    dataset_dict['label'].append(label)
                    dataset_dict['country'].append(country)
                    dataset_dict['id'].append(item['id'])  # Use the ID from the item
                    dataset_dict['instruction_id'].append(inst_idx)  # Add instruction index for reproducibility
                    dataset_dict['dataset_name'].append('countryrc')  # Add dataset name for identification
                    dataset_dict['options'].append(options_str)  # Store the shuffled options as strings

        # Create a Dataset from the dictionary
        dataset_processed = Dataset.from_dict(dataset_dict)
        datasets.append(dataset_processed)

    # CommonsenseQA dataset
    if 'commonsenseqa' in dataset_names:
        assert target_countries is None, "target_countries must be None for CommonsenseQA dataset."
        assert target_data == 'all', "target_data must be 'all' for CommonsenseQA dataset."
        instructions = NEURON_PROMPTS['commonsenseqa']

        dataset = load_dataset('tau/commonsense_qa', split='validation')

        dataset_dict = {
            'input_text': [],
            'input_ids': [],
            'attention_mask': [],
            'label': [],
            'country': [],
            'id': [],
            'instruction_id': [],
            'dataset_name': [],
            'options': [],
        }
        for inst_idx, instruction in enumerate(instructions):
            for item in dataset:
                question = item['question']
                choices = item['choices']
                answer_key = item['answerKey']
                # Create options
                options = {l: c for l, c in zip(choices['label'], choices['text'])}
                input_text = instruction.format(
                    question=question,
                    option_a=options['A'],
                    option_b=options['B'],
                    option_c=options['C'],
                    option_d=options['D'],
                    option_e=options['E'],
                )
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
                label = answer_key  # The answerKey is already in the format of 'A', 'B', 'C', 'D', or 'E'
                dataset_dict['input_text'].append(input_text)
                dataset_dict['input_ids'].append(tokenized['input_ids'][0])
                dataset_dict['attention_mask'].append(tokenized['attention_mask'][0])
                dataset_dict['label'].append(label)
                dataset_dict['country'].append('')  # No country information in CommonsenseQA
                dataset_dict['id'].append(str(item['id']))
                dataset_dict['instruction_id'].append(inst_idx)  # Add instruction index for reproducibility
                dataset_dict['dataset_name'].append('commonsenseqa')  # Add dataset name for identification
                dataset_dict['options'].append(['A', 'B', 'C', 'D', 'E'])  # Add options for the question
        # Create a Dataset from the dictionary
        dataset_processed = Dataset.from_dict(dataset_dict)
        datasets.append(dataset_processed)

    # QNLI dataset
    if 'qnli' in dataset_names:
        dataset = load_dataset('nyu-mll/glue', 'qnli', split='validation')  # Use validation split for QNLI
        assert target_countries is None, "target_countries must be None for QNLI dataset."
        assert target_data == 'all', "target_data must be 'all' for QNLI dataset."
        instructions = NEURON_PROMPTS['qnli']
        dataset_dict = {
            'input_text': [],
            'input_ids': [],
            'attention_mask': [],
            'label': [],
            'country': [],
            'id': [],
            'instruction_id': [],
            'dataset_name': [],
            'options': [],
        }
        for inst_idx, instruction in enumerate(instructions):
            for item in dataset:
                question = item['question']
                sentence = item['sentence']
                label = item['label']
                input_text = instruction.format(question=question, sentence=sentence)
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
                dataset_dict['input_text'].append(input_text)
                dataset_dict['input_ids'].append(tokenized['input_ids'][0])
                dataset_dict['attention_mask'].append(tokenized['attention_mask'][0])
                dataset_dict['label'].append(label)
                dataset_dict['country'].append('')  # No country information in QNLI
                dataset_dict['id'].append(str(item['idx']))  # Use the index as ID
                dataset_dict['instruction_id'].append(inst_idx)  # Add instruction index for reproducibility
                dataset_dict['dataset_name'].append('qnli')  # Add dataset name for identification
                dataset_dict['options'].append(['0', '1'])  # QNLI has two options: '0' for not entailment, '1' for entailment
        # Create a Dataset from the dictionary
        dataset_processed = Dataset.from_dict(dataset_dict)
        datasets.append(dataset_processed)

    # MRPC dataset
    if 'mrpc' in dataset_names:
        dataset = load_dataset('nyu-mll/glue', 'mrpc', split='test')
        assert target_countries is None, "target_countries must be None for MRPC dataset."
        assert target_data == 'all', "target_data must be 'all' for MRPC dataset."
        instructions = NEURON_PROMPTS['mrpc']
        dataset_dict = {
            'input_text': [],
            'input_ids': [],
            'attention_mask': [],
            'label': [],
            'country': [],
            'id': [],
            'instruction_id': [],
            'dataset_name': [],
            'options': [],
        }
        for inst_idx, instruction in enumerate(instructions):
            for item in dataset:
                sentence1 = item['sentence1']
                sentence2 = item['sentence2']
                label = item['label']
                input_text = instruction.format(sentence1=sentence1, sentence2=sentence2)
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
                dataset_dict['input_text'].append(input_text)
                dataset_dict['input_ids'].append(tokenized['input_ids'][0])
                dataset_dict['attention_mask'].append(tokenized['attention_mask'][0])
                dataset_dict['label'].append(label)
                dataset_dict['country'].append('')  # No country information in MRPC
                dataset_dict['id'].append(str(item['idx']))
                dataset_dict['instruction_id'].append(inst_idx)  # Add instruction index for reproducibility
                dataset_dict['dataset_name'].append('mrpc')  # Add dataset name for identification
                dataset_dict['options'].append(['0', '1'])  # MRPC has two options: '0' for not paraphrase, '1' for paraphrase
        # Create a Dataset from the dictionary
        dataset_processed = Dataset.from_dict(dataset_dict)
        datasets.append(dataset_processed)

    # Concatenate all processed datasets
    if len(datasets) == 0:
        raise ValueError("No datasets were loaded. Please check the dataset names and target countries.")
    dataset = concatenate_datasets(datasets)

    # Create a DataLoader
    def collator(batch):
        input_texts = [item['input_text'] for item in batch]
        input_ids = pad_sequence([torch.tensor(item['input_ids']) for item in batch], batch_first=True, padding_value=tokenizer.pad_token_id, padding_side='left')
        attention_mask = pad_sequence([torch.tensor(item['attention_mask']) for item in batch], batch_first=True, padding_value=0, padding_side='left')
        labels = [item['label'] for item in batch]
        countries = [item['country'] for item in batch]
        ids = [item['id'] for item in batch]
        instruction_ids = [item['instruction_id'] for item in batch]
        dataset_names = [item['dataset_name'] for item in batch]
        options = [item['options'] for item in batch]
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels, 'countries': countries,
                'input_texts': input_texts, 'ids': ids, 'instruction_ids': instruction_ids, 'dataset_names': dataset_names, 'options': options}

    if torch.distributed.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=True)
    else:
        sampler = None
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collator, sampler=sampler, shuffle=False, pin_memory=True)

    return dataloader


if __name__ == "__main__":
    # Example usage
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    dataloader = load_dataset_neuron_scores(
        dataset_names=['normadcontrol'],
        tokenizer=tokenizer,
        batch_size=4,
        target_countries=None,
        target_data='neuron',
    )

    country_counts = {}
    for batch in dataloader:
        for country in batch['countries']:
            if country not in country_counts:
                country_counts[country] = 0
            country_counts[country] += 1

    print("Country counts in the dataset:")
    for country, count in country_counts.items():
        print(f"{country}: {count} samples")
    print(f'Total samples: {len(dataloader.dataset)}')

    print(f'Total batches: {len(dataloader)}')
    for batch in dataloader:
        print('Input text: ', batch['input_texts'][0])
        print('Attention mask: ', batch['attention_mask'][0])
        print('Decoded input IDs: ', tokenizer.decode(batch['input_ids'][0], skip_special_tokens=False))
        break
