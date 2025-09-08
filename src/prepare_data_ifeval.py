from datasets import load_dataset
import pandas as pd
from simple_parsing import parse
from dataclasses import dataclass
import pickle as pkl
import os
import random

random.seed(42)

# from prompts import prompt_templates
from typing import Literal, Union


@dataclass
class Args:
    output: str = "tmp/data.pkl"
    split: Literal["train"] = "train"
    max_examples: Union[int, None] = None
    shuffle: bool = False
    random_seed: int = 42
    type: Literal["structured", "unstructured"] = "structured"


# We want a fair extraction and not just randomness


def main(args: Args):
    dataset = load_dataset("google/IFEval")

    df = pd.DataFrame(dataset["train"])
    # df['answers'] = df['answers'].apply(lambda x: x['text'])

    def extract_prompt_data(df, max_n=float("inf")):
        prompts = []
        answers = []
        for idx, row in df.iterrows():
            if idx >= max_n:
                print(f"Extracted {max_n} examples {idx}")
                break

            prompt = row["prompt"]
            key = row["key"]

            if args.type == "structured":
                prompt = f'{prompt}\nResponse with a json object with the following key: "answer". For example {{"answer": "example answer"}}.'

            prompt = prompt

            prompts.append((prompt, key))

        return prompts

    max_n = args.max_examples if args.max_examples is not None else float("inf")

    df = df.sample(frac=1, random_state=args.random_seed).reset_index()

    prompts = [(idx, prompt, answer) for idx, (prompt, answer) in enumerate(extract_prompt_data(df, max_n=max_n))]

    print(prompts[0][1])
    print(prompts[0][2])
    print(len(prompts))

    folder = os.path.dirname(args.output)
    os.makedirs(folder, exist_ok=True)
    with open(args.output, "wb") as f:
        pkl.dump(prompts, f)

    # print(len(prompts))


if __name__ == "__main__":
    args = parse(Args)
    main(args)
