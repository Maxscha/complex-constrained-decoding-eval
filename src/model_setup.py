from vllm import SamplingParams

MODEL_CONFIG = {
    "llama2": {
        "model_name": "meta-llama/Llama-2-7b-hf", 
        # "template": """[INST]
        # <<SYS>>
        # <</SYS>>

        # {prompt} [/INST]""",
        "template": "{prompt}",
        "sampling_parameters": SamplingParams(temperature=0, min_tokens=3, max_tokens=512, repetition_penalty=1.5)
    },
    "llama2-chat": {
        "model_name": "meta-llama/Llama-2-7b-chat-hf", 
        "template": """[INST]
        <<SYS>>
        <</SYS>>

        {prompt} [/INST]""",
        "sampling_parameters": SamplingParams(temperature=0, min_tokens=0, max_tokens=512)
    },
    
    "code-llama":
        {
        "model_name": "meta-llama/CodeLlama-7b-hf", 
        "template": """[INST]
        <<SYS>>
        <</SYS>>

        {prompt} [/INST]""",
        "sampling_parameters": SamplingParams(temperature=0, min_tokens=0, max_tokens=512)
    },
    "code-llama-instruct":
        {
        "model_name": "meta-llama/CodeLlama-7b-Instruct-hf", 
        "template": """[INST]
        <<SYS>>
        <</SYS>>

        {prompt} [/INST]""",
        "sampling_parameters": SamplingParams(temperature=0, min_tokens=0, max_tokens=512)
    },
    
    "llama3-base": {
        "model_name": "meta-llama/Meta-Llama-3-8B",
        "template": """<|start_header_id|>user<|end_header_id|>
        
        {prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        
        """,
        "sampling_parameters": SamplingParams(temperature=0, min_tokens=0, max_tokens=512)
    },
    
    "llama3" : {
        "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
        "template": """<|start_header_id|>user<|end_header_id|>
    
    {prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    
    """,
        "sampling_parameters": SamplingParams(temperature=0, min_tokens=0, max_tokens=512)
    },
    
    "llama31-base" : {
        "model_name": "meta-llama/Meta-Llama-3.1-8B",
        "template": """<|start_header_id|>user<|end_header_id|>
    
    {prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    
    """,
        "sampling_parameters": SamplingParams(temperature=0, min_tokens=0, max_tokens=512)
    },
    
    
    "llama31-instruct" : {
        "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "template": """<|start_header_id|>user<|end_header_id|>
    
    {prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    
    """,
        "sampling_parameters": SamplingParams(temperature=0, min_tokens=0, max_tokens=512)
    },
    
    "llama31-instruct-adapted": {
        "model_name": "./models/llama3-adapted",
        "template": "<|user|>\n{prompt}<|end|>\n<|assistant|>\n",
        "sampling_parameters": SamplingParams(temperature=0, min_tokens=0, max_tokens=512),
        "tokenizer": "meta-llama/Meta-Llama-3.1-8B"
    },
    
    
    "phi3": {
        "model_name": "microsoft/Phi-3-mini-4k-instruct",
        "template": "<|user|>\n{prompt}<|end|>\n<|assistant|>\n",
        "sampling_parameters": SamplingParams(temperature=0, min_tokens=0, max_tokens=512)
    },
    "phi4": {
        "model_name": "microsoft/phi-4",
        "template": "<|im_start|>user<|im_sep|>{prompt}<|im_end|><|im_start|>assistant<|im_sep|>",
        "sampling_parameters": SamplingParams(temperature=0, min_tokens=0, max_tokens=512)
    },
    
    "phi3-adapted": {
        "model_name": "./models/phi-3-adapted-short",
        "template": "<|user|>\n{prompt}<|end|>\n<|assistant|>\n",
        "sampling_parameters": SamplingParams(temperature=0, min_tokens=0, max_tokens=512),
        "tokenizer": "microsoft/Phi-3-mini-4k-instruct"
    },
    
    "mistral": {
        "model_name": "mistralai/Mistral-7B-v0.3",
        "template": "[INST] {prompt} [/INST]",
        "sampling_parameters": SamplingParams(temperature=0, min_tokens=0, max_tokens=512)
    },
    "mistral-instruct": {
        "model_name": "mistralai/Mistral-7B-Instruct-v0.3",
        "template": "[INST] {prompt}[/INST]",
        # "sampling_parameters": SamplingParams(temperature=0, min_tokens=0, max_tokens=512)
        "sampling_parameters": SamplingParams(temperature=0.3, min_tokens=0, max_tokens=512)
    },
    
    "models/custom_models/mistral_3_baseline": {
        "model_name": "./models/custom_models/mistral_3_baseline",
        "template": "[INST] {prompt} [/INST]",
        "sampling_parameters": SamplingParams(temperature=0, min_tokens=0, max_tokens=512),
        "tokenizer": "mistralai/Mistral-7B-v0.3"
    },
    "models/custom_models/mistral_3_strong_penalty": {
        "model_name": "./models/custom_models/mistral_3_strong_penalty",
        "template": "[INST] {prompt} [/INST]",
        "sampling_parameters": SamplingParams(temperature=0, min_tokens=0, max_tokens=512),
        "tokenizer": "mistralai/Mistral-7B-v0.3"
    },
    
    "olmo": {
        "model_name": "allenai/OLMo-7B-hf",
        "template": "{prompt}",
        "sampling_parameters": SamplingParams(temperature=0, min_tokens=0, max_tokens=512)
    },
    "olmo-instruct": {
        "model_name": "ssec-uw/OLMo-7B-Instruct-hf",
        "template": "<|endoftext|>\n\n<|user|>\n{prompt}\n\n\n<|assistant|>\n\n",
        "sampling_parameters": SamplingParams(temperature=0, min_tokens=0, max_tokens=512)
    },
    
    "deepseek": {
        "model_name": "deepseek-ai/deepseek-llm-7b-base",
        "template": "{prompt}",
        "sampling_parameters": SamplingParams(temperature=0, min_tokens=0, max_tokens=512)
    },
    
    "deepseek-chat": {
        "model_name": "deepseek-ai/deepseek-llm-7b-chat",
        "template": """User: {prompt}
        
        Assistant:""",
        "sampling_parameters": SamplingParams(temperature=0, min_tokens=0, max_tokens=512)
    },
    
    "deepseek-coder": {
        "model_name": "deepseek-ai/deepseek-coder-7b-base-v1.5",
        "template": "{prompt}",
        "sampling_parameters": SamplingParams(temperature=0, min_tokens=0, max_tokens=512)
    },
    
    "deepseek-coder-instruct": {
        "model_name": "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
        "template": """### Instruction: 
        {prompt}
        ### Response:""",
        "sampling_parameters": SamplingParams(temperature=0, min_tokens=0, max_tokens=512)
    },
    
    "qwen": {
        "model_name": "Qwen/Qwen2.5-7B",
        "template": """<|im_start|>system
                    <|im_end|>
                    <|im_start|>user
                    {prompt}<|im_end|>
                    <|im_start|>assistant
                    """,
        "sampling_parameters": SamplingParams(temperature=0, min_tokens=0, max_tokens=512)
        # "sampling_parameters": SamplingParams(temperature=0.8, top_p=0.95, min_tokens=0, max_tokens=512)
    },
    "qwen-instruct": {
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "template": """<|im_start|>system
                    <|im_end|>
                    <|im_start|>user
                    {prompt}<|im_end|>
                    <|im_start|>assistant
                    """,
        "sampling_parameters": SamplingParams(temperature=0, min_tokens=0, max_tokens=512)
        # "sampling_parameters": SamplingParams(temperature=0.8, top_p=0.95, min_tokens=0, max_tokens=512)
    },
    
    "gemma": {
        "model_name": "google/gemma-3-4b-pt",
        "template": """<bos><start_of_turn>user


        {prompt}<end_of_turn>
        <start_of_turn>model
        """,
        "sampling_parameters": SamplingParams(temperature=0, min_tokens=0, max_tokens=512)
    },
    "gemma-instruct": {
        "model_name": "google/gemma-3-4b-it",
        "template": """<bos><start_of_turn>user
        
        
        {prompt}<end_of_turn>
        <start_of_turn>model
        """,
        "sampling_parameters": SamplingParams(temperature=0, min_tokens=0, max_tokens=512)
    },
    "amber-instruct": {
        "model_name": "LLM360/AmberChat",
        "template": """{{ .System }}
            USER: {{prompt}}
            ASSISTANT: """,
        "sampling_parameters": SamplingParams(temperature=0, min_tokens=0, max_tokens=512),
    },
    "olmo2-instruct": {
        "model_name": "allenai/OLMo-2-1124-7B-Instruct",
        "template": "<|endoftext|><|user|>\n{prompt}\n<|assistant|>\n",
        "sampling_parameters": SamplingParams(temperature=0, min_tokens=0, max_tokens=512)
    },
    "vicuna-instruct": {
        "model_name": "lmsys/vicuna-7b-v1.5",
        "template": """USER: {prompt} ASSISTANT:""",
        "sampling_parameters": SamplingParams(temperature=0, min_tokens=0, max_tokens=512)
    },
    
    "qwen3-instruct": {
        "model_name": "Qwen/Qwen3-8B",
        "template": "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        "sampling_parameters": SamplingParams(temperature=0, min_tokens=0, max_tokens=512)
    },
    "smollm3": {
        "model_name": "HuggingFaceTB/SmolLM3-3B",
        "template": """<|im_start|>system
## Metadata

Knowledge Cutoff Date: June 2025
Today Date: 10 July 2025
Reasoning Mode: /no_think

## Custom Instructions

You are a helpful AI assistant named SmolLM, trained by Hugging Face.

<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
<think>

</think>
        """,
        "sampling_parameters": SamplingParams(temperature=0, min_tokens=0, max_tokens=512)
    }
}