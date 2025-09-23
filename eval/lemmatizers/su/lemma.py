import argparse
import pandas as pd
import unicodedata
from tqdm import tqdm

# git clone https://github.com/setiawanirwan/SUSTEM.git & cp SundaRootWordVer20220216.txt to current directory
from SUSTEM.SUSTEM_S import EcsStemmer


LANG = 'su'
COUNTRY = ['West Java']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, help='Input CSV file with results')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    stemmer = EcsStemmer()
    df = pd.read_csv(args.input_file)
    new_results = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        if row['country'] not in COUNTRY:
            continue
        answer = row["answers"]
        # answer is a string of python list
        answer = eval(answer)
        try:
            answer_lemmas = [[unicodedata.normalize('NFKC', token.lower()) for token in stemmer.stemmingProcess(ans.replace('(','').replace(')',''))] for ans in answer]
        except Exception as e:
            print(f"Error processing answer: {answer}, error: {e}")
            continue

        prediction = row["prediction"]
        try:
            pred_lemmatized = [unicodedata.normalize('NFKC', token.lower()) for token in stemmer.stemmingProcess(prediction.replace('(','').replace(')',''))]
        except Exception as e:
            print(f"Error processing prediction: {prediction}, error: {e}")
            continue

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
