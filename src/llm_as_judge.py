from vllm import LLM, SamplingParams
import pickle as pkl
from dataclasses import dataclass
from simple_parsing import parse
import json
import os
from squad.prompts import prompt_templates

@dataclass
class Args:
    input_path: str
    output_path: str
    num_gpus: int = 1
    multi: bool = False

def is_json(myjson):
    try:
        json_object = json.loads(myjson)
    except ValueError as e:
        return False
    return True

def is_json_with_answer(myjson):
    try:
        json_object = json.loads(myjson)
        if "answer" in json_object.keys():
            return True
    except ValueError as e:
        return False
    return False

def flatten_and_parse(input_data, output_data):
    references = []
    predictions = []
    
    for i in range(len(input_data)):
        if not is_json(output_data[i].outputs[0].text):
            # print('no json')
            continue
            
        output = json.loads(output_data[i].outputs[0].text)
        
        if i == 0:
            print(output)
        
        for idx in range(len(output.values())):
            # print(len(output.values()), len(input_data[i][0]))
            if f"question{idx + 1}" in output.keys() and idx < len(input_data[i][0]):
                references.append(output[f"question{idx + 1}"])
                predictions.append(input_data[i][0][idx])
        
    print(len(references), len(predictions))
    return references, predictions
    
    
def is_json(text):
    try:
        r = json.loads(text)
        # r["answer"]
        return True
    except:
        return False

def main(args: Args):
    
    print("Input path: ", args.input_path)
    print("Output path: ", args.output_path)

    is_struct = False

    
    # model, generation_type, prompt_id , k = args.output_path.split('/')[-1].split('_')
    if args.multi:
        model, generation_type, prompt_id  = args.output_path.split('/')[-1].split('_')
    else:
        model, generation_type, prompt_id, _ = args.output_path.split('/')[-1].split('_')
    
    if args.multi:
        null_template = None
    else:
        null_template = prompt_templates[f'{generation_type}_{prompt_id}']['null_template']
    

    if "structured" == generation_type:
        is_struct = True

    llm = LLM('meta-llama/Meta-Llama-3.1-8B-Instruct', tensor_parallel_size=args.num_gpus, max_model_len=8000)
    # output_path = '/workspaces/structure-induced-hallucination/data/output/output_llama2_unstructured_5_1.pkl'
    input_path = args.input_path
    output_path = args.output_path
    
    with open(output_path, 'rb') as f:
        data = pkl.load(f)
        data = [request for idx, request in data]
        
    with open(input_path, 'rb') as f:
        input_data = pkl.load(f)

    input_data = [(answers, prompt) for idx, prompt, answers in input_data]

    stop_signs = ["Is Correct: Yes", "Is Correct: YES", "Is Correct:  Yes", "Is Correct: No", "Is Correct: NO", "Is Correct:  No"]

    params = SamplingParams(
        # repetition_penalty=1.5,
        temperature=0,
        stop=stop_signs,
        include_stop_str_in_output=True,
        max_tokens=512
    )

    prompt_format = """
    You will be given a system answer, and reference answers tuple.
    Your task is to provide a 'is_correc' based on the system answer and reference answer. 

    Give your answer either as "Yes" or "No", depending if its correctly answered or not. The task is extractive question answering, but the form of the answer can be different. It is okay for the system answer to be embedded in a sentence, as long as the answer is correct.
    Some questions don't have an answer. This is indicated by the [], in the reference answers.  In that case the system answer should be {null_template}.
    All references answers are correct, but the system answer only needs to fit to one of the reference answers.


    Provide your feedback as follows:

    Feedback:::
    Evaluation: (your rationale for the rating, as a text)
    Is Correct: (Yes or No)

    You MUST provide values for 'Evaluation:' and 'Is Correct:' in your answer.

    Now here are the context, question and answer.

    Reference Answers: {reference}
    System Answer: {answer}

    Provide your feedback.
    Feedback:::
    Evaluation: """

    template ="""<|start_header_id|>user<|end_header_id|>
        
    {prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        
    """

    from tqdm import tqdm
    errors = 0
    accuracy = 0


    res = []
    text_results = []

    
    
    if args.multi:
        references, predictions = flatten_and_parse(input_data, data)
        len_references = sum([len(input[0]) for input in input_data])
    else:
        references = [i[0] for i in input_data]
        predictions = [out.outputs[0].text for out in data]
        len_references = len(references)
        
    if is_struct and not args.multi:
        references_predictions = [(r,json.loads(p)["answer"]) if is_json_with_answer(p) else "N/A" for r,p in   zip(references, predictions) if is_json(p)]
        references, predictions = zip(*references_predictions)
        
    
    
        
        
        

    prompts = []

    
    # print(len(data))
    
    # exit()
    
    for i in tqdm(range(0, len(references))):
        # text = data[i].outputs[0].text
        # if is_struct and not args.multi:
        #     if is_json(text):
        #         text = json.loads(text)["answer"]
        #     else:
        #         continue
        # prompt = prompt_format.format(reference = input_data[i][0], answer = text, null_template=null_template)
        prompt = prompt_format.format(reference = references[i], answer = predictions[i], null_template=null_template)

        prompt = template.format(prompt=prompt)
        prompts.append(prompt)
    
    print(prompt[0])
        
    results = llm.generate(prompts, params)
        
    for i in tqdm(range(0, len(predictions))):
        text = results[i].outputs[0].text
        
        text_results.append(text)
        cleaned_result = text.strip().split('\n')[-1]


        if cleaned_result.strip() not in stop_signs:
            print(cleaned_result)
            # print(result[0]['generated_text'])
            errors += 1
            continue

        res.append(cleaned_result)
        
        if cleaned_result.strip().split(' ')[-1] == 'Yes':
            accuracy += 1

    accuracy = accuracy / len_references

    print(f"Errors: {errors}")
    print(f"Accuracy: {accuracy}")
    
    with open("results.txt", "a") as f:
        f.write("\n" + args.output_path)
        f.write("\nErrors: " + str(errors))
        f.write("\nAccuracy: " + str(accuracy))
        f.write("\n" + str(len(predictions)))
    
    os.makedirs('data/llm_judge_output', exist_ok=True)
    
    
    with open(f'data/llm_judge_output/{model}_{generation_type}_output.pkl', 'wb') as f:
        pkl.dump(text_results, f)
    
    
if __name__ == "__main__":
    args = parse(Args)
    main(args)