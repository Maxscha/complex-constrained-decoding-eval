from datasets import load_dataset
import pandas as pd
from simple_parsing import parse
from dataclasses import dataclass
import pickle as pkl
import os
from copy import deepcopy
import random
from squad.prompts import prompt_templates
from typing import Literal, Union
import json

random.seed(42)

@dataclass
class Args:
    output: str = 'tmp/data.pkl'
    template: str = 'structured_1'
    k_shot_examples: int = 4
    split: Literal["validation", "train"]= 'validation'
    max_examples: Union[int, None] = None
    shuffle: bool = False
    random_seed: int = 42

# We want a fair extraction and not just randomness

def extract_even(question_answers, n):
    assert n % 2 == 0
    
    num_answerable = sum([1 for _, answer in question_answers if len(answer) > 0])
    num_non_answerable = len(question_answers) - num_answerable
    
    n_answerable = n // 2
    n_non_answerable = n // 2
        
    if n_answerable > num_answerable:
        n_answerable = num_answerable
        n_non_answerable = n - n_answerable
    elif n_non_answerable > num_non_answerable:
        n_non_answerable = num_non_answerable
        n_answerable = n - n_non_answerable
    
    
    cleaned_question_answers = deepcopy(question_answers)
    
    answerable_questions = [(question, answer) for question, answer in cleaned_question_answers if len(answer) > 0]
    non_answerable_questions = [(question, answer) for question, answer in cleaned_question_answers if len(answer) == 0]
    
    random.shuffle(answerable_questions)
    random.shuffle(non_answerable_questions)
    
    answerable_questions_selected = answerable_questions[:n_answerable]
    non_answerable_questions_selected = non_answerable_questions[:n_non_answerable]
    
    left_over_questions = answerable_questions[n_answerable:] + non_answerable_questions[n_non_answerable:]
    random.shuffle(left_over_questions)
    
    return answerable_questions_selected + non_answerable_questions_selected, left_over_questions
    
    
    

def extract_few_shot_fair(question_answers, max_n=float('inf')):
    if len(question_answers) == 0:
        return []
    if len(question_answers) <= max_n:
        return question_answers
    
    if max_n % 2 == 0:
        selected_questions, _ = extract_even(question_answers, max_n)
    else:
        max_n -= 1
        selected_questions, left_over_questions = extract_even(question_answers, max_n)
        selected_questions.append(left_over_questions.pop())
    
    random.shuffle(selected_questions)
    
    return selected_questions
        
        
        
        
        
    # if its an even number we take n/2 with answers and n/2 without check before if possible



schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "answer",
    "type": "object",
    "required" : ["answer"],
    "properties": {
        "answer": {
            "type": ["string", "null"],
        }
    }
}
    
schema = json.dumps(schema)


def main(args: Args):
    dataset = load_dataset("rajpurkar/squad_v2")[args.split]

    df = pd.DataFrame(dataset)
    df['answers'] = df['answers'].apply(lambda x: x['text'])

    def extract_prompt_data(df, max_n=float('inf')):
        prompts = []
        answers = []
        for idx, row in df.iterrows():
            if idx >= max_n:
                print(f"Extracted {max_n} examples {idx}")
                break
            
            title = row['title']
            text = row['context']
            question = row['question']
            answer = row['answers']
            
            same_context = df[(df['context'] == row['context']) & (df['question'] != row['question'])]
            
            other_questions = same_context['question'].values
            other_answers = same_context['answers'].values
            
            other_questions_answers = list(zip(other_questions, other_answers))
            
            other_questions_answers = extract_few_shot_fair(other_questions_answers, args.k_shot_examples)
            
            prompt_template = prompt_templates[args.template]
            
            try:
                prompt = prompt_template['base'].format(title=title, context=text, question=question, schema=schema)  # noqa: F524

                for other_question, other_answer in other_questions_answers:
                    if len (other_answer) == 0:
                        other_answer = prompt_template['null_template']
                    else:
                        selected_answer = random.sample(other_answer, 1)[0]
                        other_answer = prompt_template['answer_template'].format(answer=selected_answer)
                    prompt += prompt_template['examples'].format(question=other_question, answer=other_answer)
                
                prompt += prompt_template['suffix'].format(question=question)
            except Exception as e:
                print(f"Error with {idx}")
                print(e)
                print(prompt_template)
                raise e
                
            prompts.append((prompt, list(set(answer))))
            
        return prompts
    
    max_n = args.max_examples if args.max_examples is not None else float('inf')
    
    df = df.sample(frac=1, random_state=args.random_seed).reset_index()
        
    prompts = [(idx, prompt, answer) for idx, (prompt, answer) in enumerate(extract_prompt_data(df, max_n=max_n))]
    
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

    