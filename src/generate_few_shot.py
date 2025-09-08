from simple_parsing import parse
from dataclasses import dataclass, field
import pickle as pkl
import vllm
from vllm.inputs import TokensPrompt
from model_setup import MODEL_CONFIG
import os
from tqdm import tqdm
from typing import Optional
from transformers import AutoTokenizer

@dataclass
class Args:
    model: str = 'llama3'
    model_path: Optional[str] = None
    input_files: list[str] = field(default_factory=list)
    output_files: list[str] = field(default_factory=list)
    schema_type: str = 'default'
    use_logits_processor: bool = False
    max_tokens: Optional[int] = None
    logprobs: Optional[int] = None
    
    
    def __post_init__(self):
        assert len(self.input_files) == len(self.output_files), "The number of input files must match the number of output files."
    
    
schema_types = {
    'default': """
    {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "answer",
        "type": "object",
        "properties": {
            "answer": {
              "anyOf": [
                    {
                        "type": "string"
                    },
                    {
                        "type": "null"
                    }
                ]
            }
        }
    }
    """,
    'summary': """
    {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "summary",
        "type": "object",
        "required" : ["summary"],
        "properties": {
            "summary": {
                "type": "string"
            }
        }
    }
    """,
    'squad_multi': """
    {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "answer",
        "type": "object",
        "required" : ["question1", "question2", "question3", "question4", "question5", "question6", "question7", "question8"],
        "properties": {
            "question1": {
                "anyOf": [
                    {
                        "type": "string"
                    },
                    {
                        "type": "null"
                    }
                ]
            },
            "question2": {
                "anyOf": [
                    {
                        "type": "string"
                    },
                    {
                        "type": "null"
                    }
                ]
            },
            "question3": {
                "anyOf": [
                    {
                        "type": "string"
                    },
                    {
                        "type": "null"
                    }
                ]
            },
            "question4": {
                "anyOf": [
                    {
                        "type": "string"
                    },
                    {
                        "type": "null"
                    }
                ]
            },
            "question5": {
                "anyOf": [
                    {
                        "type": "string"
                    },
                    {
                        "type": "null"
                    }
                ]
            },
            "question6": {
                "anyOf": [
                    {
                        "type": "string"
                    },
                    {
                        "type": "null"
                    }
                ]
            },
            "question7": {
                "anyOf": [
                    {
                        "type": "string"
                    },
                    {
                        "type": "null"
                    }
                ]
            },
            "question8": {
                "anyOf": [
                    {
                        "type": "string"
                    },
                    {
                        "type": "null"
                    }
                ]
            }
        }
    }
    """,
    
    'thinking': """    
    {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "answer",
    "type": "object",
    "required" : ["thoughts", "answer"],
    "properties": {
        "thoughts": {
            "type": "string"
        },
        "answer": {
            "anyOf": [
                    {
                        "type": "string"
                    },
                    {
                        "type": "null"
                    }
                ]
        }
    }
    }""",
    
    'reasoning': """    
    {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "answer",
    "type": "object",
    "required" : ["reasoning", "answer"],
    "properties": {
        "reasoning": {
            "type": "string"
        },
        "answer": {
            "type": "integer"
        }
    }
    }""",
    
    'factscore': """    
    {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "answer",
    "type": "object",
    "required" : ["bio", "birthplace", "occupation", "birthyear"],
    "properties": {
        "bio": {
            "type": "string"
        },
        "birthplace": {
            "type": "string"
        },
        "occupation": {
            "type": "string"
        },
        "birthyear": {
            "type": "integer"
        }
    }
    }""",

}
def add_logits_processor(sampling_params, tokenizer, schema_type='default'): 
    import vllm.model_executor.guided_decoding.outlines_logits_processors
    
    schema = schema_types[schema_type]

    sampling_params.logits_processors=[vllm.model_executor.guided_decoding.outlines_logits_processors.JSONLogitsProcessor(schema, tokenizer,None, None)]

def main(args: Args):
    print(args)
    config = MODEL_CONFIG[args.model] 
    
    if "tokenizer" in config:
        tokenizer = config['tokenizer']
        print(f"Using tokenizer from config {tokenizer}")
    else:
        tokenizer = config['model_name']
        print(f"Using tokenizer from model {tokenizer}")
    
    if args.model_path is not None:
        model_path = args.model_path
    else:
        model_path = config['model_name']
    
        
    # TODO overwork LM-Format-Enforcer
    llm = vllm.LLM(model=model_path, trust_remote_code=True, guided_decoding_backend="lm-format-enforcer", tokenizer=tokenizer)
    
    tokenizer_instance = AutoTokenizer.from_pretrained(tokenizer)
    
    sampling_parameters = config['sampling_parameters']
    if args.max_tokens is not None:
        sampling_parameters.max_tokens = args.max_tokens
    
    if args.logprobs is not None:
        sampling_parameters.logprobs = args.logprobs
        
    if args.use_logits_processor:
        add_logits_processor(sampling_parameters, llm.get_tokenizer(), schema_type=args.schema_type)
        
        
    for input_file, output_file in zip(args.input_files, args.output_files):
        # if output_file already exist skip
        if os.path.exists(output_file):
            continue
    
    
        with open(input_file, 'rb') as f:
            ids, prompt_conversations, _ = zip(*pkl.load(f))
        if args.schema_type == 'thinking':        
            raise ValueError("Thinking schema type is not supported yet")
        
        tokens = [tokenizer_instance.apply_chat_template(conversation,tokenize=True, add_generation_prompt=True) for conversation in prompt_conversations]
        
        prompts = [TokensPrompt(prompt_token_ids=token_ids) for token_ids in tokens]
        
        # templated_prompts = [config['template'].format(prompt=prompt) for prompt in text_prompts]
    
        
        def batch(lst, n):
            for i in range(0, len(lst), n):
                yield lst[i:i + n]
    
    
        results = []
        batch_size = 8
        for b in tqdm(batch(prompts, batch_size), total=len(prompts) // batch_size):
            try:
                llm_results = llm.generate(b, sampling_parameters, use_tqdm=False)
            except Exception as e:
                llm_results = []
                for prompt in b:
                    try:
                        llm_results.append(llm.generate(prompt, sampling_parameters, use_tqdm=False)[0])
                    except Exception as e:
                        print(f"Failed to generate for prompt: {prompt}")
                        
                        llm_results.append(None)
                
            
            results.extend(llm_results)
        results = [(idx, result) for idx, result in  enumerate(results)]
    
        folder = os.path.dirname(output_file)
        os.makedirs(folder, exist_ok=True)
    
        with open(output_file, 'wb') as f:
            pkl.dump(results, f)    

if __name__ == "__main__":
    args = parse(Args)
    main(args)

    
