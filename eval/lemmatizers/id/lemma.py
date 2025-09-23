import argparse
import pandas as pd
from tqdm import tqdm
import unicodedata

from nlp_id.lemmatizer import Lemmatizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, help='Input CSV file with results')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    df = pd.read_csv(args.input_file)

    nlp = Lemmatizer()

    new_results = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        if row['country'] not in ['Indonesia']:
            continue
        answer = row["answers"]
        # answer is a string of python list
        answer = eval(answer)
        answer_lemmas = []
        for ans in answer:
            res = nlp.lemmatize(ans)
            lemmas = []
            for lemma in res.split():
                lemmas.append(unicodedata.normalize('NFKC', lemma).lower())
            answer_lemmas.append(lemmas)

        prediction = row["prediction"]
        res = nlp.lemmatize(prediction)
        pred_lemmatized = [unicodedata.normalize('NFKC', lemma).lower() for lemma in res.split()]

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
    output_file = args.input_file.replace('.csv', '_lemmatized_id.csv')
    new_df.to_csv(output_file, index=False)
    acc = new_df['is_correct'].mean()
    print(f"Accuracy (lemmatized, id): {acc:.4f}")


if __name__ == "__main__":
    main()
