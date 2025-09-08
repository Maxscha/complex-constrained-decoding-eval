import pickle as pkl
import json
import pandas as pd
# from squad.prompts import prompt_templates
import glob
from dataclasses import dataclass
from simple_parsing import parse
import os
import re
import string


@dataclass
class Args:
    data_path: str = "data/squad_val_multi/42"
    fuzzy_threshold: float = 0.05


def read_experiment(model, type, prompt_id, few_shot="", data_path="data"):
    input_path = f"{data_path}/input/{type}_{prompt_id}.pkl"
    output_path = f"{data_path}/output/{model}_{type}_{prompt_id}.pkl"
    
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

    # template = prompt_templates[f"{type}_{prompt_id}"]

    return input, predictions, None


def is_json(myjson):
    try:
        json_object = json.loads(myjson)
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
    test_references = [normalize_answer(reference) for reference in references if len(normalize_answer(reference)) > 0]
    prediction = normalize_answer(prediction)
    if len(prediction) == 0:
        return 0, 0, 0
    if len(test_references) == 0:
        print("test")
        return 0, 0, 0
    
    references = test_references
    
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
    
    

def get_scores(prediction_references, none_value=None):
    precision, recall, f1 = zip(
        *[precision_recall_f1(prediction, references, none_value) for prediction, references in prediction_references]
    )
    precision = sum(precision) / len(precision)
    recall = sum(recall) / len(recall)
    f1 = sum(f1) / len(f1)
    
    en_in = sum([get_en_in(prediction, references, none_value) for prediction, references in prediction_references]) / len(prediction_references)

    return precision, recall, f1, en_in

def flatten(xss):
    return [x for xs in xss for x in xs]

def eval(model, type, prompt_id, data_path="data", fuzzy_threshold=0.05):
    input, predictions, null_template = read_experiment(
        model, type, prompt_id, data_path=data_path
    )
    
    if len(input) == 0:
        return None

    # Only if structured
    if type == "structured":
        cleaned = [(json.loads(r), a) for r, a in zip(predictions, input) if is_json(r)]
        is_json_percentage = len(cleaned) / len(predictions)
        
        
        answer_keys = [f"question{i}" for i in range(1, 9)]
        
        has_answer_key = len([r for r, a in cleaned if any([k in r.keys() for k in answer_keys])]) / len(cleaned)
        
        
        prediction_references = [zip(prediction.values(), references) for prediction, references in cleaned]
        
        prediction_references = flatten(prediction_references)
            
        
        # prediction_references = [
        #     (r["answer"], [str(answer).strip() for answer in a]) for r, a in cleaned if "answer" in r.keys()
        # ]
        none_value = None
    elif type == "unstructured":
        assert False, "Unstructured not implemented"
        prediction_references = [(r.strip(), [str(answer).strip() for answer in a]) for r, a in zip(predictions, input)]
        none_value = null_template
        is_json_percentage = 0
        has_answer_key = 0

    precision, recall, f1, en_in = get_scores(prediction_references, none_value=none_value)

    no_answer = [(r, a) for r, a in prediction_references if len(a) == 0]
    no_answer_precision, no_answer_recall, no_answer_f1, no_answer_en_in = get_scores(no_answer, none_value=none_value)

    with_answer = [(r, a) for r, a in prediction_references if len(a) > 0]
    with_answer_precision, with_answer_recall, with_answer_f1, with_answer_en_in = get_scores(with_answer, none_value=none_value)

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
    assert all([len(c) == 3 for c in categories])
    models, types, prompt_ids = zip(*[(c[0], c[1], c[2]) for c in categories])

    return list(set(models)), list(set(types)), list(set(prompt_ids))
# , list(set(k))


def eval_all(args: Args):
    path = args.data_path
    result_dict = []
    MODELS, TYPES, PROMPT_IDS = get_categories(path)

    MODELS = sorted(MODELS)

    # K = [f"_{k}" for k in K]

    TYPES = sorted(TYPES)
    MODELS = sorted(MODELS)
    PROMPT_IDS = sorted(PROMPT_IDS)
    # K = sorted(K)

    for type in TYPES:
        # for n in K:
        for model in MODELS:
            for prompt_id in PROMPT_IDS:
                try:
                    print(f"Model: {model}, Prompt: {prompt_id} Type: {type}")
                    assert prompt_id != 1
                    result = eval(model, type, prompt_id, data_path=path)
                    
                    result = result | {"model": model, "prompt_id": prompt_id, "type": type}
                    result_dict.append(result)
                    print(f"Total EN_IN Match: {result['total_en_in']}")
                except Exception as e:
                    print(e)
                    continue
    df = pd.DataFrame(result_dict)

    outputh_path = "results"
    if not os.path.exists(outputh_path):
        os.makedirs(outputh_path)

    df.to_pickle(f"{outputh_path}/results_{path.replace('/', '')}.pkl")


if __name__ == "__main__":
    args = parse(Args)
    eval_all(args)
