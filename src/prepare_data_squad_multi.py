from datasets import load_dataset
import pandas as pd
from simple_parsing import parse
from dataclasses import dataclass
import pickle as pkl
import os
from copy import deepcopy
import random
from squad.prompts_multi import prompt_templates
from typing import Literal, Union
import json
from tqdm import tqdm
import numpy as np
random.seed(42)

@dataclass
class Args:
    output: str = 'tmp/data.pkl'
    split: Literal["validation", "train"]= 'validation'
    template: str = 'structured_1'
    max_examples: Union[int, None] = None
    shuffle: bool = False
    random_seed: int = 42

# We want a fair extraction and not just randomness


def get_few_shot_examples(df, context, n=1):
    """
    Get random few shot examples from a dataframe, excluding the given context.
    Optimized version that avoids expensive operations.
    
    Parameters:
    df (pd.DataFrame): Input dataframe with columns [context, title, question, answers]
    context (str): Context to exclude from examples
    n (int): Number of examples to return (currently only supports n=1)
    
    Returns:
    list: List of dictionaries containing few shot examples
    """
    assert n == 1
    
    # Create mask for filtering instead of creating new DataFrame
    mask = df['context'] != context
    
    # Early return if no valid contexts
    if not mask.any():
        return []
    
    # Get unique contexts efficiently using numpy
    # unique_contexts = df.loc[mask, 'context'].values
    # unique_contexts = np.unique(unique_contexts)
    
    # Sample one context
    selected_context = df.loc[mask, 'context'].sample(1).iloc[0]
    
    # Get all rows for selected context efficiently
    context_mask = df['context'] == selected_context
    context_data = df[context_mask]
    
    # Create single example dictionary without groupby
    example = {
        'context': selected_context,
        'title': context_data['title'].iloc[0],
        'question': context_data['question'].tolist(),
        'answers': context_data['answers'].tolist()
    }
    
    return [example]

def generate_answer_example(answers):
    answers = [answer[0] if len(answer) > 0 else None for answer in answers]
    answers = {f"question{idx+1}": answer for idx, answer in enumerate(answers)}
    return json.dumps(answers, indent=4)

def generate_schema_and_example(num_questions):
    key_schema = "question{idx}"
    
    # up to12
    test_answers = ["Answer", "Other Answer", None, "More Other Answers", None, "Different Answer", None, "Different Answers", "Other Answers", "Other Answer", None, "More Other Answers", None, "Different Answer", None, "Different Answers", "Other Answers", "Other Answer", None, "More Other Answers", None, "Different Answer", None, "Different Answers", "Other Answers", "Other Answer", None, "More Other Answers", None, "Different Answer", None, "Different Answers", "Other Answers", "Other Answer", None, "More Other Answers", None, "Different Answer", None, "Different Answers", "Other Answers", "Other Answer", None, "More Other Answers", None, "Different Answer", None, "Different Answers", "Other Answers", "Other Answer", None, "More Other Answers", None, "Different Answer", None, "Different Answers", "Other Answers"]
    
    test_answers = test_answers[:num_questions]
    
    keys = [key_schema.format(idx=idx) for idx in range(1, num_questions+1)]
    
    value_schema = json.loads("""
    {
        "anyOf": [
            {
                "type": "string"
            },
            {
                "type": "null"
            }
        ]
    }
    """)
    
    schema = {key: value_schema for key in keys}
    
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "answer",
        "type": "object",
        "required" : keys,
        "properties": schema
    }
    schema = json.dumps(schema)
    
    print(num_questions)
    example = {key: test_answers[idx] for idx, key in enumerate(keys)}
    example = json.dumps(example, indent=4)
    example = example.replace("\n", "\n    ")
    # example = example.replace(" ", " ")
    

    return schema, example

def main(args: Args):
    dataset = load_dataset("rajpurkar/squad_v2")[args.split]

    df = pd.DataFrame(dataset)
    df['answers'] = df['answers'].apply(lambda x: x['text'])

    def extract_prompt_data(df, max_n=float('inf')):
        prompts = []
        answers = []
        
        for groups in tqdm(df.groupby('context')):
            title = groups[1]['title'].values[0]
            context = groups[0]
            questions = groups[1]['question'].values
            answers = groups[1]['answers'].values
            
            if len(questions) > 10:
                continue
            
            # Problem is that the num of questions are not the same, so lets skip the enforced thing for now
            # assert len(questions) == 9 or len(questions) == 10 or len(questions) == 8
            
            schema, example = generate_schema_and_example(len(questions))
        

            # few_shot_example = get_few_shot_examples(df, context)

            questions = enumerate(questions)
            question_text = "\n".join([f"{idx+1}: {question}" for idx, question in questions])
            
            template = "structured_3"
            
            prompt_template = prompt_templates[template]
            
            prompt = prompt_template['base'].format(title=title, context=context, example=example, question_text=question_text, schema=schema)
            
            # for example in few_shot_example:
                # example_questions_text = "\n".join([f"{idx+1}: {question}" for idx, question in enumerate(example['question'])])
                # example_context = example['context']
                # example_answers = generate_answer_example(example['answers'])
                # prompt += prompt_template['examples'].format(context=example_context, questions=example_questions_text, answers=example_answers)
                
            max_n -= len(answers)
            
            prompt += prompt_template['suffix'].format(question_text=question_text)

            

            prompts.append((prompt, answers))
            
            if max_n <= 0:
                break
            
        return prompts
    max_n = args.max_examples if args.max_examples is not None else float('inf')
    
    df = df.sample(frac=1, random_state=args.random_seed).reset_index()
        
    prompts = [(idx, prompt, answer) for idx, (prompt, answer) in tqdm(enumerate(extract_prompt_data(df, max_n=max_n)))]
    
    print(prompts[0][1])
    print(prompts[0][2])
    print(len(prompts))
    


    folder = os.path.dirname(args.output)
    os.makedirs(folder, exist_ok=True)
    with open(args.output, 'wb') as f:
        pkl.dump(prompts, f)        

if __name__ == "__main__":
    args = parse(Args)
    main(args)

    