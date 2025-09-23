import argparse
import pandas as pd
import spacy
import unicodedata
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, help='Input CSV file with results')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    nlp = spacy.load("en_core_web_sm")
    df = pd.read_csv(args.input_file)
    new_results = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        if row['country'] not in ['USA', 'UK']:
            continue
        answer = row["answers"]
        # answer is a string of python list
        answer = eval(answer)
        answer_lemmas = [[unicodedata.normalize('NFKC', token.lemma_.lower()) for token in nlp(ans)] for ans in answer]

        prediction = row["prediction"]
        pred_lemmatized = [unicodedata.normalize('NFKC', token.lemma_.lower()) for token in nlp(prediction)]

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
    output_file = args.input_file.replace('.csv', '_lemmatized_en.csv')
    new_df.to_csv(output_file, index=False)
    acc = new_df['is_correct'].mean()
    print(f"Accuracy (lemmatized, en): {acc:.4f}")


if __name__ == "__main__":
    main()
