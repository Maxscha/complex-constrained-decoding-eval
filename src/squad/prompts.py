import json

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

prompt_templates = { 
    "structured_1" :
    {
        "base": "Answer each question using information in the preceding background paragraph. If there is not enough informationen provided answer with, answer with null. The answer should be returned in json-format like {{\"answer\" :\"str\"|null}} such as the following example: {{\"answer\": \"example answer\"}}.\n\nTitle: {title} \n\nBackground: {context}",
        "null_template": "null",
        "answer_template": "\"{answer}\"",
        "examples": "\n\nQuestion: {question} Answer: {{\"answer\" : {answer}}}",
        "suffix": "\n\nQuestion: {question} Answer: "
    },
    "structured_2" :
    {
        "base": "Answer each question using information in the preceding background paragraph. The answer should be returned in json-format like {{\"answer\" :\"str\"}} or {{\"answer\" : null}} if there is not enough information. See the following for an example: {{\"answer\": \"example answer\"}}.\n\nTitle: {title} \n\nBackground: {context}",
        "null_template": "null",
        "answer_template": "\"{answer}\"",
        "examples": "\n\nQuestion: {question} Answer: {{\"answer\" : {answer}}}",
        "suffix": "\n\nQuestion: {question} Answer: "
    },
    "structured_3" :
    {
        "base": "Answer each question using information in the preceding background paragraph. If there is not enough informationen provided answer with, answer with null. The answer should be returned in json-format like {{\"answer\" :\"str\"|null}}. See the following for an example: {{\"answer\": \"example answer\"}}.\n\nTitle: {title} \n\nBackground: {context}\n\n",
        "null_template": "null",
        "answer_template": "\"{answer}\"",
        "examples": "\n\nQuestion: {question}\nAnswer: {{\"answer\" : {answer}}}",
        "suffix": "Question: {question}\nAnswer: "
    },
    "structured_4" : {
        "base": "Answer each question using information in the preceding background paragraph. The answer has to be returned in json:{{\"answer\" :\"str\"|null}}, like the following example: {{\"answer\": \"example answer\"}}.\n\nBackground: {context}",
        "null_template": "null",
        "answer_template": "\"{answer}\"",
        "examples": "\n\nQuestion: {question}\nAnswer: {{\"answer\" : {answer}}}",
        "suffix": "\n\nQuestion: {question}\nAnswer: "
    },
    "structured_5" : {
        "base": "Answer each question using information in the preceding background paragraph. Use JSON:{{\"answer\" :\"str\"|null}}. See the following for an example: {{\"answer\": \"example answer\"}}.\n\nBackground: {context}",
        "null_template": "null",
        "answer_template": "\"{answer}\"",
        "examples": "\n\n{question}\n{{\"answer\" : {answer}}}",
        "suffix": "\n\n{question}\n"
    },
    "structured_6" : {
        "base": "Using the following context:\n{context}\nAnswer the last question. Use the following json format:{{\"answer\" :\"str\"|null}}. See the following for an example: {{\"answer\": \"example answer\"}}.\nHere are some concrete examples:",
        "null_template": "null",
        "answer_template": "\"{answer}\"",
        "examples": "\n{question}\n{{\"answer\" : {answer}}}",
        "suffix": "\n{question}\n"
    },
    "structured_7" : {
        "base": "Using the following context:\n{context}\nAnswer the question. Use json for answering:{{\"answer\" :\"str\"|null}}. See the following for an example: {{\"answer\": \"example answer\"}}.\nHere are some examples:",
        "null_template": "null",
        "answer_template": "\"{answer}\"",
        "examples": "\n\n{question}\n{{\"answer\" : {answer}}}",
        "suffix": "\nHere is the question:\n{question}\n"
    },
    "structured_8" : {
        "base": "Using the following context:\n{context}\nAnswer the question as a json-object with the following schema: {{\"answer\" :\"str\"|null}}. If there is no answer use null. See the following for an example: {{\"answer\": \"example answer\"}}.\nIn the following you see some examples:",
        "null_template": "null",
        "answer_template": "\"{answer}\"",
        "examples": "\n\n{question}\n{{\"answer\" : {answer}}}",
        "suffix": "\n\nHere is the question:\n{question}\nAnswer:"
    },

    # UNSTRUCTURED
    
    "unstructured_1" : {
        "base": "Answer each question using information in the preceding background paragraph. If there is not enough information provided, answer with 'Not in background.'\n\nTitle: {title} \n\nBackground: {context}",
        "null_template": "Not in background.",
        "answer_template": "{answer}",
        "examples": "\n\nQuestion: {question} Answer: {answer}",
        "suffix": "\n\nQuestion: {question} Answer: "
    },
    "unstructured_2" : {
        "base": "Answer each question using information in the preceding background paragraph. If there is not enough information to answer the question, answer with 'Not in background.'\n\nTitle: {title} \n\nBackground: {context}",
        "null_template": "Not in background.",
        "answer_template": "{answer}",
        "examples": "\n\nQuestion: {question} Answer: {answer}",
        "suffix": "\n\nQuestion: {question} Answer: "
    },
    "unstructured_3" : {
        "base": "Answer each question using information in the preceding background paragraph. If there is not enough information to answer the question, answer with 'Can't answer'\n\nTitle: {title} \n\nBackground: {context}",
        "null_template": "Can't answer",
        "answer_template": "{answer}",
        "examples": "\n\nQuestion: {question}\nAnswer: {answer}",
        "suffix": "\n\nQuestion: {question}\nAnswer: "
    },
    "unstructured_4" : {
        "base": "Answer the last question using information in the preceding background paragraph. If there is not enough information to answer the question, answer with 'Can't answer'\n\nBackground: {context}",
        "null_template": "Can't answer",
        "answer_template": "{answer}",
        "examples": "\n\nQuestion: {question}\nAnswer: {answer}",
        "suffix": "\n\nQuestion: {question}\nAnswer: "
    },
    "unstructured_5" : {
        "base": "Answer each question using information in the preceding background paragraph. If there is not enough information to answer the question, answer with 'Can't answer'\n\nBackground: {context}",
        "null_template": "Can't answer",
        "answer_template": "{answer}",
        "examples": "\n\n{question}\n{answer}",
        "suffix": "\n\n{question}\n"
    },
    "unstructured_6" : {
        "base": "Using the following context:\n{context}\nAnswer the last question. If you can't answer the question write 'Can't answer'.\nHere are some examples:",
        "null_template": "Can't answer",
        "answer_template": "{answer}",
        "examples": "\n{question}\n{answer}",
        "suffix": "\n{question}\n"
    },
    "unstructured_7" : {
        "base": "Using the following context:\n{context}\nAnswer the question. If you can't answer the question write 'Can't answer'.\nHere are some examples:",
        "null_template": "Can't answer",
        "answer_template": "{answer}",
        "examples": "\n{question}\n{answer}",
        "suffix": "\nHere is the question:\n{question}\n"
    },
    "unstructured_8" : {
        "base": "Using the following context:\n{context}\nAnswer the question. If there is no answer to the question write 'No answer'.\nIn the following you see some examples:",
        "null_template": "No answer",
        "answer_template": "{answer}",
        "examples": "\n\nQuestion:{question}\nAnswer:{answer}",
        "suffix": "\n\nHere is the question:\n{question}\nAnswer:"
    },

    "structured_9" : {
        "base": "Answer the following question(s) based exclusively on the provided context. If the answer is not evident from the context answer with the keyword null. Use JSON format for the answer. See the following for an example: {{\"answer\": \"example answer\"}}.\n{context}",
        "examples": "\nQuestion: {question}\nAnswer: {{\"answer\": {answer}}}",
        "answer_template": "\"{answer}\"",
        "null_template": "null",
        "suffix": "\nQuestion: {question}\nAnswer: "
    },

    "unstructured_9" : {
        "base": "Answer the following question(s) based exclusively on the provided context. If the answer is not evident from the context answer with the keyword null.\n{context}",
        "answer_template": "{answer}",
        "null_template": "null",
        "examples": "\nQuestion: {question}\nAnswer: {answer}",
        "suffix": "\nQuestion: {question}\nAnswer: "
    },

    "structured_10" : {
        "base": "Considering the following context {context}, answer the following questions. Use JSON as output fprmat. Do only use information from the context. If the answer to the question is not in the context, set the answer value to null. See the following for an example: {{\"answer\": \"example answer\"}}.\nContext:\nTitle: {title}\n{context}\nSome Examples:",
        "examples": "\nQuestion: {question}\nAnswer: {{\"answer\": {answer}}}",
        "answer_template": "\"{answer}\"",
        "null_template": "null",
        "suffix": "\nQuestion: {question}\nAnswer: "
    },

    "unstructured_10" : {
        "base": "Considering the following context {context}, answer the following questions. Do only use information from the context. If the answer to the question is not in the context, answer with null. Some Examples:\nContext:\nTitle: {title}\n",
        "answer_template": "{answer}",
        "null_template": "null",
        "examples": "\nQuestion: {question}\nAnswer: {answer}",
        "suffix": "\nQuestion: {question}\nAnswer: "
    },

    "structured_11" : {
        "base": "Answer the questions about this text:\n{context}\n If the answer is not in the text output null . Use JSON format. See the following for an example: {{\"answer\": \"example answer\"}}.",
        "examples": "\nQuestion: {question}\nAnswer: {{\"answer\": {answer}}}",
        "answer_template": "\"{answer}\"",
        "null_template": "null",
        "suffix": "\nQuestion: {question}\nAnswer: {{\"answer\": "
    },

    "unstructured_11" : {
        "base": "Answer the questions about this text:\n{context}\n If the answer is not in the text output null ",
        "answer_template": "{answer}",
        "null_template": "null",
        "examples": "\nQuestion: {question}\nAnswer: {answer}",
        "suffix": "\nQuestion: {question}\nAnswer: "
    },
    
    "structured_12" : {
        "base": "This is an extractive question answering task. Given a context and a question, you should extract an *exact text span* from the context that answers the question. The answers must be an excerpt from the context and must not be freely written. The answer must be structured as a JSON with the following structure: {{\"answer\": \"<answer>\"}}. See the following for an example: {{\"answer\": \"example answer\"}}. If the answer cannot be found in the context, write 'null' instead. \n\nContext: {context}",
        "examples": "\n\nExtract the answer for the following question from the context or answer with 'null'.\nQuestion: {question}\n{{\"answer\": {answer} }}",
        "answer_template": "\"{answer}\"",
        "null_template": "null",
        "suffix": "\n\nExtract the answer for the following question from the context or answer with 'null'.\nQuestion: {question}\n"
    },
    
    "unstructured_12" : {
        "base": "This is an extractive question answering task. Given a context and a question, you should extract an *exact text span* from the context that answers the question. The answers must be an excerpt from the context and must not be freely written. If the answer cannot be found in the context, write 'null' instead. \n\nContext: {context}",
        "answer_template": "{answer}",
        "null_template": "null",
        "examples": "\n\nExtract the answer for the following question from the context or answer with 'null'.\nQuestion: {question}\nAnswer: {answer}",
        "suffix": "\n\nExtract the answer for the following question from the context or answer with 'null'.\nQuestion: {question}\nAnswer:"
    },
    
    "structured_13" : {
            # This is the start of the prompt
            "base": "Imagine you know the following context:\n{context}\nThen answer the question {question}. Use a structured json format for answering, e.g., {{\"answer\" :[\"str\"] | null }}. See the following for an example: {{\"answer\": \"example answer\"}}. You can provide multiple answers if you are unsure. \nHere are some examples:",
             # These are a single example template. These are between the prefix and the suffix
             "examples": "\n\n{question}\n{{\"answer\" : [{answer}]}}",
             # This is how the example answer is templated. Note the " since we want to output a json string here
             "answer_template": "[\"{answer}\"]",
             # This is how an example without a valid answer is templated
             "null_template": "null",
             "suffix": ""

             },
        # UNSTRUCTURED

        "unstructured_13" : {
            "base": "You know that {context}",
            "answer_template": "The answer is {answer}",
            "null_template": "I do not know",
            "examples": "\n\nCan you answer {question}",
            "suffix": "\nCan you answer {question}"
             }
}