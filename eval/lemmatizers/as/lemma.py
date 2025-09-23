import argparse
import pandas as pd
import os
import unicodedata
import sys
from tqdm import tqdm

# git clone https://github.com/anoopkunchukuttan/indic_nlp_library.git & https://github.com/anoopkunchukuttan/indic_nlp_resources.git
# The path to the local git repo for Indic NLP library
INDIC_NLP_LIB_HOME=os.path.abspath("./indic_nlp_library")
# The path to the local git repo for Indic NLP Resources
INDIC_NLP_RESOURCES=os.path.abspath("./indic_nlp_resources")

sys.path.append(INDIC_NLP_LIB_HOME)
from indicnlp import common
from indicnlp import loader
from indicnlp.tokenize import indic_tokenize  


LANG = 'as'
COUNTRY = ['Assam']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, help='Input CSV file with results')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    common.set_resources_path(INDIC_NLP_RESOURCES)
    loader.load()
    df = pd.read_csv(args.input_file)
    new_results = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        if row['country'] not in COUNTRY:
            continue
        answer = row["answers"]
        # answer is a string of python list
        answer = eval(answer)
        answer_lemmas = [[unicodedata.normalize('NFKC', token.lower()) for token in indic_tokenize.trivial_tokenize(ans)] for ans in answer]

        prediction = row["prediction"]
        pred_lemmatized = [unicodedata.normalize('NFKC', token.lower()) for token in indic_tokenize.trivial_tokenize(prediction)]

        # judge if any of the answer lemmas is in the prediction lemmas
        is_correct = False
        for ans_lemmas in answer_lemmas:
            for i in range(0, len(pred_lemmatized) - len(ans_lemmas) + 1):
                idx = 0
                for j in range(len(ans_lemmas)):
                    if pred_lemmatized[i + j] != ans_lemmas[j]:
                        break
                    idx += 1
                if idx == len(ans_lemmas):
                    is_correct = True
                    break
            if is_correct:
                break

        new_results.append({
            'instruction_id': row['instruction_id'],
            'id': row['id'],
            'country': row['country'],
            'prediction': pred_lemmatized,
            'answers': answer_lemmas,
            'is_correct': is_correct,
        })

    new_df = pd.DataFrame(new_results)
    output_file = args.input_file.replace('.csv', f'_lemmatized_{LANG}.csv')
    new_df.to_csv(output_file, index=False)
    acc = new_df['is_correct'].mean()
    print(f"Accuracy (lemmatized, {LANG}): {acc:.4f}")


if __name__ == "__main__":
    main()
