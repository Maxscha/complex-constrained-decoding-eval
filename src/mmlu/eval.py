import argparse
import json
import os
import time
import pandas as pd
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from categories import categories, subcategories

from datasets import load_dataset

from model_setup import MODEL_CONFIG
 
choices = ["A", "B", "C", "D"]

schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "answer",
    "type": "object",
    "required" : ["answer"],
    "properties": {
        "answer": {
            "type": "string",
        }
    }
}
    
schema = json.dumps(schema)

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True, generation_type="unstructured"):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        answer = df.iloc[idx, k + 1]
        match generation_type:
            case "unstructured":
                prompt += f"{answer}\n\n"
            case "structured":
                prompt += f"{{\"answer\":\"{answer}\"}}\n\n"
                
    return prompt


def gen_prompt(train_df, subject, k=-1, generation_type="unstructured"):
    if generation_type == "structured":
        extra = f" Answer with a JSON object with the following schema:{schema} such as the following example {{\"answer\": \"A\"}}."
    else:
        extra = ""
    
    
    prompt = "The following are multiple choice questions (with answers) about {}.{}\n\n".format(
        format_subject(subject),
        extra
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i, generation_type=generation_type)
    return prompt


@torch.no_grad()
def eval(args, subject, model, tokenizer, dev_df, test_df, template):
    cors = []
    all_probs = []
    answers = choices[: test_df.shape[1] - 2]

    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k, generation_type=args.type)
        prompt = train_prompt + prompt_end
        if args.type == "structured":
            prompt += "{\"answer\": \""
        
        # print(prompt)
        # exit()
        
        prompt = template.format(prompt=prompt)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

        while input_ids.shape[-1] > 2048:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k, generation_type=args.type)
            prompt = train_prompt + prompt_end
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(
                model.device
            )

        label = test_df.iloc[i, test_df.shape[1] - 1]

        logits = model(input_ids=input_ids).logits[0, -1]

        probs = (
            torch.nn.functional.softmax(
                torch.tensor(
                    [
                        logits[tokenizer("A").input_ids[-1]],
                        logits[tokenizer("B").input_ids[-1]],
                        logits[tokenizer("C").input_ids[-1]],
                        logits[tokenizer("D").input_ids[-1]],
                    ]
                ).float(),
                dim=0,
            )
            .detach()
            .cpu()
            .numpy()
        )
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]

        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs


def main(args):
    
    model_name_or_path = MODEL_CONFIG[args.model]["model_name"]
    
    tokenizer_name = model_name_or_path
    if "tokenizer" in MODEL_CONFIG[args.model]:
        tokenizer_name = MODEL_CONFIG[args.model]["tokenizer"]
        
    if args.model_path is not None:
        model_name_or_path = args.model_path
    
    template = MODEL_CONFIG[args.model]["template"]
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model.eval()
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(args.data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    results_path = os.path.join(args.save_dir, f"results_{args.model}_{args.type}")
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}

    for subject in subjects:
        dev_df = pd.read_csv(
            os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None
        )[: args.ntrain]
        test_df = pd.read_csv(
            os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None
        )

        cors, acc, probs = eval(args, subject, model, tokenizer, dev_df, test_df, template)
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

        test_df["{}_correct".format(args.model)] = cors
        for j in range(probs.shape[1]):
            choice = choices[j]
            test_df["{}_choice{}_probs".format(args.model, choice)] = probs[:, j]
        test_df.to_csv(
            os.path.join(
                results_path, "{}.csv".format(subject)
            ),
            index=None,
        )

    results = {"subcategories": {}, "categories": {}}
    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        results["subcategories"][subcat] = subcat_acc
        print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        results["categories"][cat] = cat_acc
        print("Average accuracy {:.3f} - {}".format(cat_acc, cat))
    weighted_acc = np.mean(np.concatenate(all_cors))
    # results["weighted_accuracy"] = weighted_acc
    print("Average accuracy: {:.3f}".format(weighted_acc))
    
    # todo put into dataframe

    results_file = os.path.join(
        args.save_dir, "accuracies_{}_{}.json".format(args.model.replace("/", "_"), args.type)
    )
    with open(results_file, "w") as f:
        json.dump(results, f)
    
    if args.results_df is not None:
        if not os.path.exists(args.results_df):
            results_df = pd.DataFrame(columns=["model", "type", "accuracy"])
        else:
            results_df = pd.read_pickle(args.results_df)
        results_df
        
        
        results_df = results_df._append({"model": model_name_or_path, "type": args.type, "accuracy": weighted_acc, "k": args.ntrain}, ignore_index=True)
        
        results_df.to_pickle(args.results_df)
        
        pd.set_option('display.max_columns', None)  # Show all columns
        pd.set_option('display.width', 1000)        # Set width to 1000 characters
        pd.set_option('display.max_colwidth', 100)  # Set maximum column width
        pd.set_option('display.max_rows', None)     # Show all rows
        
        print(results_df)
        
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--data_dir", "-d", type=str, default="data")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--model", "-m", type=str)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--type", "-t", type=str, default="unstructured")
    parser.add_argument("--results_df", type=str, default=None)
    args = parser.parse_args()
    main(args)