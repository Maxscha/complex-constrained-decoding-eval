import argparse


import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
from datasets import load_dataset

from model_setup import MODEL_CONFIG
 
import random

random.seed(42)
 
choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M"]

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

def format_example(df, idx, include_answer=True, generation_type="unstructured"):
    prompt = df.iloc[idx, 0]
    text_choices = df.iloc[idx, 1]    
    if generation_type == "structured":
        extra = f" Answer with a JSON object with the following schema:{schema} such as the following example {{\"answer\": \"A\"}}."
    else:
        extra = ""
    
    prompt += extra
    
    for j in range(len(text_choices)):
        prompt += "\n{}. {}".format(choices[j], text_choices[j])
    if generation_type == "unstructured":
        prompt += "\nAnswer:"
    if include_answer:
        raise NotImplementedError()
        # answer = df.iloc[idx, 2]
        # match generation_type:
            # case "unstructured":
                # prompt += f"{answer}\n\n"
            # case "structured":
                # prompt += f"{{\"answer\":\"{answer}\"}}\n\n"
    
    return prompt


def gen_prompt(train_df, k=-1, generation_type="unstructured"):    
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i, generation_type=generation_type)
    return prompt


def shuffle_choices(example):
    combined = list(zip(example['choices'], example['labels']))
    
    random.shuffle(combined)
    
    

    return {'choices': [choice for choice, _ in combined], 'labels': [label for _, label in combined]}

@torch.no_grad()
def eval(args, model, tokenizer, test_df, template):
    cors = []
    all_probs = []
    # answers = choices[: test_df.shape[1] - 2]

    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False, generation_type=args.type)
        
        prompt = prompt_end

        
        # print(prompt)
        # exit()
        
        prompt = template.format(prompt=prompt)
        
        if args.type == "structured":
            prompt += "{\"answer\": \""
        else:
            prompt += " "
        
        
        if i == 0:
            print(f'{prompt}|')
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

        label = test_df.iloc[i, test_df.shape[1] - 1]

        logits = model(input_ids=input_ids).logits[0, -1]

        probs = (
            torch.nn.functional.softmax(
                torch.tensor(
                    [logits[tokenizer(i).input_ids[-1]] for i in choices]
                    # [
                        
                    #     logits[tokenizer("A").input_ids[-1]],
                    #     logits[tokenizer("B").input_ids[-1]],
                    #     logits[tokenizer("C").input_ids[-1]],
                    #     logits[tokenizer("D").input_ids[-1]],
                    # ]
                ).float(),
                dim=0,
            )
            .detach()
            .cpu()
            .numpy()
        )
        
        dict_probs = {i:choices[i] for i in range(len(choices))}
        pred = dict_probs[np.argmax(probs)]

        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    # print("Average accuracy {:.3f} - {}".format(acc))

    return cors, acc, all_probs


def main(args):
    model_name_or_path = MODEL_CONFIG[args.model]["model_name"]
    template = MODEL_CONFIG[args.model]["template"]
    
    if args.model_path is not None:
        model_name_or_path = args.model_path
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        device_map="auto",
        trust_remote_code=True,
    )
    
    tokenizer_name = model_name_or_path
    if "tokenizer" in MODEL_CONFIG[args.model]:
        tokenizer_name = MODEL_CONFIG[args.model]["tokenizer"]
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model.eval()
    
    
    ds = load_dataset('truthfulqa/truthful_qa', 'multiple_choice')['validation']
    
    ds = ds.map(lambda x:{'choices': x['mc1_targets']['choices'], 'labels': x['mc1_targets']['labels']}, remove_columns=['mc1_targets', 'mc2_targets'])
    
    ds = ds.map(shuffle_choices)
    
    ds = ds.map(lambda x: {'labels': [chr(ord('A') + idx) for idx, label in enumerate(x['labels']) if label == 1][0]})
    
    test_df = pd.DataFrame(ds)
    
    cors, acc, probs = eval(args, model, tokenizer, test_df, template)
    
    print(acc)
    
    if args.results_df is not None:
        if not os.path.exists(args.results_df):
            results_df = pd.DataFrame(columns=["model", "type", "accuracy"])
        else:
            results_df = pd.read_pickle(args.results_df)
        results_df
        
        
        results_df = results_df._append({"model": model_name_or_path, "type": args.type, "accuracy": acc}, ignore_index=True)
        
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