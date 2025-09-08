import pprint

lines = []

with open('output.txt') as f:
    lines = f.readlines()
    
def parse_accuracy(accuracy_lines):
    prompt_level_accuracy = float(accuracy_lines[0].split(': ')[1])
    instruction_level_accuracy = float(accuracy_lines[1].split(': ')[1])
    
    
    # only last 3 digits
    return {
        'prompt_level_accuracy': f"{prompt_level_accuracy:.3f}",
        'instruction_level_accuracy': f"{instruction_level_accuracy:.3f}",
    }

    
lines = [l.strip() for l in lines if l.strip() != '']
results = {}

for i, line in enumerate(lines):
    if line.endswith('eval_results_loose.jsonl Accuracy Scores:'):
        model_type = line.split('/')[2]
        results[model_type] = parse_accuracy(lines[i+1:i+3])


pprint.pprint(results)
