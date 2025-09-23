import argparse
import pandas as pd
from tqdm import tqdm
import unicodedata

import sparknlp
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import Tokenizer, LemmatizerModel
from pyspark.ml import Pipeline
from sparknlp.base import LightPipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, help='Input CSV file with results')
    args = parser.parse_args()
    return args


def nlp_pipeline():
    spark = sparknlp.start()

    # 1. DocumentAssembler
    document_assembler = DocumentAssembler() \
        .setInputCol("text") \
        .setOutputCol("document")

    # 2. Tokenizer
    tokenizer = Tokenizer() \
        .setInputCols(["document"]) \
        .setOutputCol("token")

    # 3. Lemmatizer
    lemmatizer = LemmatizerModel.pretrained("lemma", "am") \
        .setInputCols(["token"]) \
        .setOutputCol("lemma")

    # 4. Pipeline
    nlp_pipeline = Pipeline(stages=[document_assembler, tokenizer, lemmatizer])

    # 5. LightPipeline
    empty_df = spark.createDataFrame([['']]).toDF("text")
    light_pipeline = LightPipeline(nlp_pipeline.fit(empty_df))
    return light_pipeline


def main():
    args = parse_args()
    df = pd.read_csv(args.input_file)

    nlp = nlp_pipeline()

    new_results = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        if row['country'] not in ['Ethiopia']:
            continue
        answer = row["answers"]
        # answer is a string of python list
        answer = eval(answer)
        answer_lemmas = []
        for ans in answer:
            res = nlp.fullAnnotate(ans)
            lemmas = []
            for token, lemma in zip(res[0]["token"], res[0]["lemma"]):
                lemmas.append(unicodedata.normalize('NFKC', lemma.result.lower()))
            answer_lemmas.append(lemmas)

        prediction = row["prediction"]
        res = nlp.fullAnnotate(prediction)
        pred_lemmatized = [unicodedata.normalize('NFKC', lemma.result.lower()) for lemma in res[0]["lemma"]]

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
    output_file = args.input_file.replace('.csv', '_lemmatized_am.csv')
    new_df.to_csv(output_file, index=False)
    acc = new_df['is_correct'].mean()
    print(f"Accuracy (lemmatized, am): {acc:.4f}")


if __name__ == "__main__":
    main()
