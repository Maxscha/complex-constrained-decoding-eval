import pickle as pkl

import json

from simple_parsing import parse
from dataclasses import dataclass

@dataclass
class Args:
    input_path: str = '/workspaces/structure-induced-hallucination/data/ifeval/42/input/unstructured.pkl'
    output_path: str = '/workspaces/structure-induced-hallucination/data/ifeval/42/output/llama31-instruct_structured.pkl'
    input_replacement: str = '/workspaces/structure-induced-hallucination/src/ifeeval/data/input_data.jsonl'
    output: str = '/workspaces/structure-induced-hallucination/src/ifeeval/data/input_response_data_llama31-instruct_structured.jsonl'

def convert_to_dict(path):
    prompt_id_to_prompt = {}
    with open(path, 'r') as f:
        f = f.readlines()
        prompt_replacement = [json.loads(l) for l in f]
        for l in prompt_replacement:
            prompt_id_to_prompt[l['key']] = l['prompt']
    
    return prompt_id_to_prompt

def is_json_with_answer(response):
    try:
        json.loads(response)["answer"]
        return True
    except:
        return False

def main(args: Args):
    # output_path = '/workspaces/structure-induced-hallucination/data/ifeval/42/output/llama3_structured.pkl'

    # input_path = '/workspaces/structure-induced-hallucination/data/ifeval/42/input/structured.pkl'

    # input_replacement = '/workspaces/structure-induced-hallucination/src/ifeeval/data/input_data.jsonl'

    prompt_id_to_prompt = convert_to_dict(args.input_replacement)

    with open(args.input_path, 'rb') as f:
        input_data = pkl.load(f)
    
    if args.output_path.endswith('_unstructured.pkl'):
        output_type = 'unstructured'
    elif args.output_path.endswith('_structured.pkl'):
        output_type = 'structured'
    else:
        raise ValueError("Invalid input path")
    
    with open(args.output_path, 'rb') as f:
        output_data = pkl.load(f)


    _, prompt, prompt_id = zip(*input_data)
    _, response = zip(*output_data)

    responses = [r.outputs[0].text for r in response]

    if output_type == 'structured':
        
        responses = [json.loads(r)["answer"] if is_json_with_answer(r) else r for r in responses]



    combined = list(zip(prompt, responses, prompt_id))


    with open(args.output, 'w') as f:
        for prompt, response, prompt_id in combined:
            
            f.write(json.dumps({
                'prompt': prompt_id_to_prompt[prompt_id],
                'response': response,
                'prompt_id': prompt_id
            }))
            f.write('\n')
        
        
        
        

if __name__ == "__main__":
    args = parse(Args)
    main(args)

