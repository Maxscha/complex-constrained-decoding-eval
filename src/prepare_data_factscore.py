from datasets import load_dataset
import pandas as pd
from simple_parsing import parse
from dataclasses import dataclass
import pickle as pkl
import os
from copy import deepcopy
import random

random.seed(42)

# from prompts import prompt_templates
from typing import Literal, Union

@dataclass
class Args:
    output: str = 'tmp/data.pkl'
    split: Literal["train"]= 'train'
    max_examples: Union[int, None] = None
    shuffle: bool = False
    random_seed: int = 42
    type: Literal['structured', 'unstructured'] = 'structured'

# We want a fair extraction and not just randomness



def main(args: Args):
    with open('src/prompt_entities_cleaned.txt', 'r') as f:
        entities = f.readlines()
    entities = [entity.strip() for entity in entities if entity.strip() != '']
    
    assert len(entities) > 0, "No entities found in the file."
    
    prompts = []
    
    for idx, entity in enumerate(entities):
        prompt = f"Question: Tell me a bio of {entity}"
        if args.type == 'structured':
            prompt = f"{prompt}. Answer with a JSON object with the following key: \"answer\", such as the following example {{\"answer\": \"bio\"}} " 
        prompts.append((idx, prompt, entity))
    
    
    print(prompts[0][1])
    print(prompts[0][2])
    print(len(prompts))
    


    folder = os.path.dirname(args.output)
    os.makedirs(folder, exist_ok=True)
    with open(args.output, 'wb') as f:
        pkl.dump(prompts, f)
        
    # print(len(prompts))

        

if __name__ == "__main__":
    args = parse(Args)
    main(args)

    