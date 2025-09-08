import pickle as pkl
import json
import pandas as pd
from prompts import prompt_templates
import glob
from dataclasses import dataclass
from simple_parsing import parse
import os
import re
import string


@dataclass
class Args:
    data_path: str = "best_prompt_val/42"
    fuzzy_threshold: float = 0.05


def read_experiment(model, type, few_shot="", prompt_id="", data_path="data"):
    output_path = f"{data_path}/output/{model}_{type}_{prompt_id}{few_shot}.pkl"
    
    prompt_id = prompt_id.replace('-encouraged', '')
    input_path = f"{data_path}/input/{type}_{prompt_id}{few_shot}.pkl"
    
    
    print(output_path)
    
    if os.path.exists(input_path) == False:
        print(f"Input file not found: {input_path}")
        return [], [], None
    if os.path.exists(output_path) == False:
        print(f"Output file not found: {output_path}")
        return [], [], None
    
    with open(input_path, "rb") as f:
        input = pkl.load(f)
        input = [answer for idx, res, answer in input]

    with open(output_path, "rb") as f:
        predictions = pkl.load(f)
        # predictions = [(res.outputs[0].text) for idx, res in predictions]

    test = [
        (input, prediction[1].outputs[0].text) for input, prediction in zip(input, predictions) if prediction[1] is not None
    ]
    input, predictions = zip(*test)

    template = prompt_templates[f"{type}_{prompt_id}"]

    return input, predictions, template["null_template"]


def is_json(myjson):
    try:
        json_object = json.loads(myjson)
        if json_object is None:
            return False
        if not isinstance(json_object, dict):
            return False
        if "answer" not in json_object.keys():
            return False
    except ValueError as e:
        return False
    return True


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s)))).split()


# TODO WHAT ABOUT NULL VALUES?
def precision_recall_f1(prediction, references, none_value=None):
    if len(references) == 0:
        if none_value is None:
            if prediction is None:
                return 1, 1, 1
            else:
                return 0, 0, 0
        else:
            references = [none_value]
    if prediction is None:
        return 0, 0, 0

    old_references = references
    old_prediction = prediction
    references = [normalize_answer(reference) for reference in references if len(normalize_answer(reference)) > 0]
    prediction = normalize_answer(prediction)
    if len(prediction) == 0:
        return 0, 0, 0
    precisions = []
    recalls = []
    f1s = []
    for reference in references:
        correct = 0
        for word in reference:
            if word in prediction:
                correct += 1
        precision = correct / len(prediction)
        recall = correct / len(reference)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return max(precisions), max(recalls), max(f1s)

def get_en_in(prediction, references, none_value=None):
    if isinstance(prediction, str):
        prediction = prediction.lower()
    references = [ref.lower() for ref in references]
    if len(references) == 0:
        if none_value is None:
            if prediction is None:
                return 1
            else:
                return 0
        else:
            references = [none_value.lower()]
    if prediction is None:
        return 0
    return 1 if any(ref in prediction for ref in references) else 0
    
    

def get_scores(prediction_references, none_value=None, total_length=None):
    if len(prediction_references) == 0:
        return 0, 0, 0, 0
    precision, recall, f1 = zip(
        *[precision_recall_f1(prediction, references, none_value) for prediction, references in prediction_references]
    )
    
    if total_length is None:
        total_length = len(prediction_references)
        
    
    precision = sum(precision) / total_length
    recall = sum(recall) /total_length
    f1 = sum(f1) / total_length
    
    en_in = sum([get_en_in(prediction, references, none_value) for prediction, references in prediction_references]) / total_length

    return precision, recall, f1, en_in


def eval(model, struct_type, prompt_id, few_shot, data_path="data", fuzzy_threshold=0.05):
    input, predictions, null_template = read_experiment(
        model, struct_type, few_shot=few_shot, prompt_id=prompt_id, data_path=data_path
    )
    
    if len(input) == 0:
        return None

    # Only if structured
    if struct_type == "structured":
        cleaned = [(json.loads(r), a) for r, a in zip(predictions, input) if is_json(r)]
        is_json_percentage = len(cleaned) / len(predictions)
        has_answer_key = len([r for r, a in cleaned if "answer" in r.keys()]) / len(predictions)
        prediction_references = [
            (r["answer"], [str(answer).strip() for answer in a]) for r, a in cleaned if "answer" in r.keys() and isinstance(r["answer"], (str, type(None)))
        ]
        none_value = None
    elif struct_type == "unstructured":
        prediction_references = [(r.strip(), [str(answer).strip() for answer in a]) for r, a in zip(predictions, input)]
        none_value = null_template
        is_json_percentage = 0
        has_answer_key = 0

    precision, recall, f1, en_in = get_scores(prediction_references, none_value=none_value, total_length=len(predictions))

    no_answer = [(r, a) for r, a in prediction_references if len(a) == 0]
    no_answer_precision, no_answer_recall, no_answer_f1, no_answer_en_in = get_scores(no_answer, none_value=none_value, total_length=len(predictions))

    with_answer = [(r, a) for r, a in prediction_references if len(a) > 0]
    with_answer_precision, with_answer_recall, with_answer_f1, with_answer_en_in = get_scores(with_answer, none_value=none_value, total_length=len(predictions))

    return {
        "total_precision": precision,
        "total_recall": recall,
        "total_f1": f1,
        "total_en_in": en_in,
        "no_answer_precision": no_answer_precision,
        "no_answer_recall": no_answer_recall,
        "no_answer_f1": no_answer_f1,
        "no_answer_en_in": no_answer_en_in,
        "with_answer_precision": with_answer_precision,
        "with_answer_recall": with_answer_recall,
        "with_answer_f1": with_answer_f1,
        "with_answer_en_in": with_answer_en_in,
        "is_json_percentage": is_json_percentage,
        "has_answer_key": has_answer_key,
    }

def get_categories(data_folder):
    data = glob.glob(f"{data_folder}/output/*.pkl")
    assert len(data) > 0
    # get filename
    categories = [d.split("/")[-1].replace(".pkl", "") for d in data]
    categories = [c.split("_") for c in categories]
    print(categories)
    assert all([len(c) == 4 for c in categories])
    models, types, prompt_ids, k = zip(*[(c[0], c[1], c[2], c[3]) for c in categories])

    return list(set(models)), list(set(types)), list(set(prompt_ids)), list(set(k))


def eval_all(args: Args):
    path = args.data_path
    result_dict = []
    MODELS, TYPES, PROMPT_IDS, K = get_categories(path)

    MODELS = sorted(MODELS)

    K = [f"_{k}" for k in K]

    for type in TYPES:
        for n in K:
            for model in MODELS:
                for prompt_id in PROMPT_IDS:
                    print(f"Model: {model}, Prompt: {prompt_id} Type: {type} K: {n}")
                    result = eval(model, type, prompt_id, n, data_path=path, fuzzy_threshold=args.fuzzy_threshold)
                    if result is None:
                        continue
                    result = result | {"model": model, "prompt_id": prompt_id, "type": type, "k_shot": n.replace("_", "")}
                    result_dict.append(result)


                    print(f"Total EN_IN Match: {result['total_en_in']} Is JSON: {result['is_json_percentage']}")
    df = pd.DataFrame(result_dict)

    outputh_path = "results"
    if not os.path.exists(outputh_path):
        os.makedirs(outputh_path)

    df.to_pickle(f"{outputh_path}/results_{path.replace('/', '')}.pkl")


if __name__ == "__main__":
    args = parse(Args)
    eval_all(args)
