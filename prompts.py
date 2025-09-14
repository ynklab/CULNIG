WORLDVALUESBENCH_SYSTEM_PROMPT = [
    'You are a person from {country}.',
    'Behave as if you are from {country}.',
]


WORLDVALUESBENCH_PROMPTS = [
    'Question: {question}\nPlease respond with a single digit only from {min_option} to {max_option}. Do not include any other text.',
    'Question: {question}\nYour answer should be a single digit between {min_option} and {max_option}. Do not add any other information.',
]


NORMAD_PROMPTS = [
    """You will be given a short story involving 2 or more characters about everyday situations. You will also be given a country of relevance. Read the story and determine if the character’s behavior aligns with the social norms of that country.
Only consider the provided context and country. Do not include any outside assumptions. Respond **only** with the number 1, 2, or 3.

country: {country}
Story: {story}
Options: {options}
""",

    """Read a short story describing everyday events with at least two characters. You'll also receive a country. Your task is to judge whether the character’s actions are in line with the cultural expectations of that country.
Use only the given story and cultural context; do not bring in any outside knowledge. Answer strictly with the number 1, 2, or 3.

country: {country}
Story: {story}
Options: {options}
""",

    """A short scenario featuring multiple characters will be provided, along with the cultural background of a specific country. Your role is to determine how well the behavior shown fits that country's social norms.
Do not make inferences beyond the given content. Only respond with 1, 2, or 3.

country: {country}
Story: {story}
Options: {options}
""",

    """Given a short daily-life story involving multiple characters, along with the country, assess whether the behavior depicted fits within the social norms of that culture.
Stick strictly to the given material without adding outside reasoning. Answer with just 1, 2, or 3.

country: {country}
Story: {story}
Options: {options}
""",
]


CULTURALBENCH_PROMPTS = [
    """To answer the following multiple-choice question, you should choose one option only among A,B,C,D. Do not output any other things.
Question: {question}
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}""",

    """Select only one option from A, B, C, or D to answer the following multiple-choice question. Do not output anything else.
Question: {question}
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}""",

    """Choose one answer among A, B, C, and D for the question below. Do not include any explanation or extra content.
Question: {question}
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}""",

    """You must answer the following question by selecting a single choice from A, B, C, or D. Do not write anything else.
Question: {question}
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}""",
]

COUNTRYRC_PROMPTS = [
    """Read the passage carefully and choose a single option from A, B, C, D to answer the question. Do not output any other text.

passage: {passage}
question: {question}
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}
""",

    """Read the following passage and question. Then, pick the most suitable answer from the four options. Only return the letter of your choice (A, B, C, or D).

passage: {passage}
question: {question}
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}
""",

    """From the information provided in the passage, choose the best answer to the question. You must select a single choice: 1, 2, 3, or 4, and do not include any other text.

passage: {passage}
question: {question}
1. {option_a}
2. {option_b}
3. {option_c}
4. {option_d}
""",

    """Determine the correct answer to the question based on the content of the passage. Respond with one of the following: 1, 2, 3, or 4. No additional text is needed.

passage: {passage}
question: {question}
1. {option_a}
2. {option_b}
3. {option_c}
4. {option_d}
""",
]

COMMONSENSEQA_PROMPTS = [
    """To answer the following multiple-choice question, you should choose one option only among A,B,C,D,E. Do not output any other things.
Question: {question}
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}
E. {option_e}""",

    """Choose one answer among A, B, C, D, and E for the question below. Do not include any explanation or extra content.
Question: {question}
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}
E. {option_e}""",

    """Pick one option only — A, B, C, D, or E — as the answer to the question below. Do not provide any additional text.
Question: {question}
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}
E. {option_e}""",

    """Please choose one and only one of the following options (A, B, C, D, or E) to answer the question. Do not add anything else.
Question: {question}
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}
E. {option_e}""",
]

QNLI_PROMPTS = [
    "Determine whether the following context sentence contains enough information to answer the question.\nQuestion: {question}\nContext: {sentence}\nRespond with:\n0 if it does (entailment)\n1 if it does not (not_entailment)\nOnly answer with 0 or 1.",

    "Classify the relationship between the following question and context.\nQuestion: {question}\nContext: {sentence}\nLabel as:\n0: entailment – the question is supported by the context\n1: not_entailment – the question is not supported by the context\nPlease respond with either 0 or 1 only.",

    "Read the question and the context.\nQuestion: {question}\nContext: {sentence}\nIf the context provides enough evidence to answer the question, return 0 (entailment).\nIf the context is insufficient or irrelevant, return 1 (not_entailment).\nYour answer should be either 0 or 1.",

    "Your task is to judge if the answer to the question can be found in the context.\nQuestion: {question}\nContext: {sentence}\nAnswer 0 for entailment, and 1 for not_entailment. Do not include any other text.",
]

MRPC_PROMPTS = [
    "Determine whether the following two sentences are paraphrases of each other in meaning.\nSentence 1: {sentence1}\nSentence 2: {sentence2}\nRespond with:\n1 – if they are paraphrases\n0 – if they are not paraphrases\nOnly answer with 0 or 1.",

    "You are given two sentences. Judge whether they express the same meaning, even if the wording is different.\nSentence 1: {sentence1}\nSentence 2: {sentence2}\nAnswer with 1 if they are paraphrases, and 0 if they are not.\nPlease respond using only 0 or 1.",

    "A paraphrase means that two sentences convey the same information using different words or structure.\nSentence 1: {sentence1}\nSentence 2: {sentence2}\nDecide whether these sentences are paraphrases.\nReturn 1 for paraphrase, 0 for not paraphrase.\nYour answer must be either 0 or 1.",

    "Compare the following two sentences. If they convey the same meaning regardless of differences in wording, classify them as paraphrases.\nSentence 1: {sentence1}\nSentence 2: {sentence2}\nRespond with:\n1 – if they are semantically equivalent (paraphrase)\n0 – if they are not semantically equivalent\nOnly use 0 or 1 as your answer.",
]


NEURON_PROMPTS = {
    'normad': NORMAD_PROMPTS,
    'culturalbench': CULTURALBENCH_PROMPTS,
    'worldvaluesbench': WORLDVALUESBENCH_PROMPTS,
    'countryrc': COUNTRYRC_PROMPTS,
    'commonsenseqa': COMMONSENSEQA_PROMPTS,
    'qnli': QNLI_PROMPTS,
    'mrpc': MRPC_PROMPTS,
}

NEURON_SYSTEM_PROMPTS = {
    'worldvaluesbench': WORLDVALUESBENCH_SYSTEM_PROMPT,
}
