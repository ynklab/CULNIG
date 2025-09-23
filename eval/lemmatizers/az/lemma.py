import argparse
import pandas as pd
import unicodedata
from tqdm import tqdm

# # git clone https://github.com/aznlp-disc/stemmer.git and cp word.txt & suffix.txt to current directory
from stemmer.stemmer import Stemmer
from string import punctuation


LANG = 'az'
COUNTRY = ['Azerbaijan']
my_stemmer = Stemmer()

def stem_words(my_text):
    my_text=my_text.replace("İ", "I")
    my_text=my_text.replace("“", "")
    my_text=my_text.replace("”", "")
    my_text=my_text.replace("'", "")
    my_text=my_text.replace('"', "")
    my_text=my_text.split()
    my_words=[]
    for word in my_text:
        my_words.append(''.join(c for c in word if (c not in punctuation) or (c == '-')))
    # Apply stemming to the list of words
    my_words = my_stemmer.stem_words(my_words)
    # Print words after stemming
    return my_words


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, help='Input CSV file with results')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    df = pd.read_csv(args.input_file)
    new_results = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        if row['country'] not in COUNTRY:
            continue
        answer = row["answers"]
        # answer is a string of python list
        answer = eval(answer)
        answer_lemmas = [[unicodedata.normalize('NFKC', token.lower()) for token in stem_words(ans)] for ans in answer]

        prediction = row["prediction"]
        pred_lemmatized = [unicodedata.normalize('NFKC', token.lower()) for token in stem_words(prediction)]

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
