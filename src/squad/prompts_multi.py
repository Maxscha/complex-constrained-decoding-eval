prompt_templates = { 
    "structured_1" :
    {
        "base": "Answer each question using information in the preceding background paragraph. Return a JSON object. If you can't answer a question use null, otherwise use a string. The result should look like the following:{example}\n\nTitle: {title}\n\nBackground: {context}\n\nQuestion: {question_text}\n\nHere is a concrete example:",
        "examples": "\n\nBackground: {context}\n\nQuestions:\n{questions}\nAnswers:\n{answers}",
        
        "suffix": "\nAnswers:\n"
    },
    "structured_2" :
    {
        "base": "Answer each question using information in the preceding background paragraph. The answer should be returned in JSON-format like {example}. If you can't answer a question use null, if you can answer the question use a string.\n\nTitle: {title} \n\nBackground: {context}\n\nHere is a concrete example:",
        "examples": "\n\nBackground: {context}\n\nQuestions:\n{questions}\nAnswers:\n{answers}",
        "suffix": "\n\nQuestions:\n{question_text}\n\nAnswer: "
    },
    "structured_3" :
    {
        "base": "Answer each question using information in the preceding background paragraph. If there is not enough informationen provided, answer with null. The answer should be returned in JSON format adhering to the following schema:\n{schema}\n Similiar to the following example:\n{example}\n\nTitle: {title} \n\nBackground: {context}",
        # "examples": "\n\nBackground: {context}\n\nQuestions:\n{questions}\nAnswers:\n{answers}",
        "suffix": "\n\nQuestions:\n{question_text}\n\nAnswer: "
    },
    "structured_4" : {
        "base": "Answer each question using information in the preceding background paragraph. The answer has to be returned in json:{{\"questionX\" :\"str\"|null}} where X is the number of the question, like the following example: {example}.\n\nBackground: {context}\n\nHere is a concrete example:",
        "examples": "\n\nBackground: {context}\n\nQuestions:\n{questions}\nAnswers:\n{answers}",
        "suffix": "\n\nQuestions:\n{question_text}\n\nAnswer: "
    },
    "structured_5" : {
        "base": "Answer each question using information in the preceding background paragraph. Use JSON:{{\"questionX\" :\"str\"|null}}, where X is the number of the question. See the following for an example: {example}.\n\nBackground: {context}\n\nHere is a concrete example:",
        "examples": "\n\nContext: {context}\n\n{questions}\n{answers}",
        "suffix": "\n\n{question_text}\n"
    },
    "structured_6" : {
        "base": "Using the following context:\n{context}\nAnswer each question. Use the following JSON format:{{\"questionX\" :\"str\"|null}}, where X is the number of the question. See the following for an example: {example}.\nHere is a concrete example:",
        "examples": "\n\nContext: {context}\n\n{questions}\n{answers}",
        "suffix": "\n{question_text}\n"
    },
    "structured_7" : {
        "base": "Using the following context:\n{context}\nAnswer the questions. Use JSON for answering:{{\"questionX\" :\"str\"|null}}, where X is the number of the question. See the following for an example: {example}.\nHere is a concrete example:",
        "examples": "\n\nContext: {context}\n\n{questions}\n{answers}",
        "suffix": "\nHere are the questions:\n{question_text}\n"
    },
    "structured_8" : {
        "base": "Using the following context:\n{context}\nAnswer the questions as a json-object with the following schema: {{\"questionX\" :\"str\"|null}}, where X is the number of the question. If there is no answer use null. See the following for an example: {example}.\nHere is a concrete example:",
        "examples": "\n\nContext: {context}\n\nQuestions:\n{questions}\nAnswer:\n{answers}",
        "suffix": "\n\nHere are the question:\n{question_text}\nAnswer:"
    },

    "structured_9" : {
        "base": "Answer the following question(s) based exclusively on the provided context. If the answer is not evident from the context answer with the keyword null. Use JSON format for the answer. See the following for an example: {example}.\n{context}\nHere is a concrete example:",
        "examples": "\n\nContext: {context}\n\nQuestions:\n{questions}\nAnswer:\n{answers}",
        "suffix": "\nQuestions: {question_text}\nAnswer: "
    },


    "structured_10" : {
        "base": "Considering the following context {context}, answer the following questions. Use JSON as output fprmat. Do only use information from the context. If the answer to the question is not in the context, set the answer value to null. See the following for an example: {example}.\nContext:\n{context}\nSome Examples:",
        "examples": "\n\nContext: {context}\n\nQuestions:\n{questions}\nAnswer:\n{answers}",
        "suffix": "\nQuestions:\n{question_text}\nAnswer:\n"
    },


    "structured_11" : {
        "base": "Answer the questions about this text:\n{context}\n If the answer is not in the text output null . Use JSON format. See the following for an example: {example}. Here are some concrete examples:",
        "examples": "\n\nText:\n{context}\n\nQuestions:\n{questions}\nAnswers:\n{answers}",
        "suffix": "\nQuestions:\n{question_text}\nAnswers:\n"
    },


    
    "structured_12" : {
        "base": "This is an extractive question answering task. Given a context and questions, you should extract an *exact text span* from the context that answers the question. The answers must be an excerpt from the context and must not be freely written. The answer must be structured as a JSON with the following structure: {{\"question1\": \"<answer>\"}}. See the following for an example: {example}. If the answer cannot be found in the context, write 'null' instead. \n\nContext: {context}\nHere are some concrete examples:",
        "examples": "\n\nText:\n{context}\n\nQuestions:\n{questions}\nAnswers:\n{answers}",
        "suffix": "\n\nExtract the answer for the following question from the context or answer with 'null'.\nQuestions: {question_text}\n"
    },

    "structured_13" : {
            # This is the start of the prompt
            "base": "Imagine you know the following context:\n{context}\nThen answer the questions:\n{question_text}\nUse a structured json format for answering, e.g., {{\"answer\" :[\"str\"] | null }}. See the following for an example: {example}.\n\n\nHere are some concrete examples:",
        "examples": "\n\nText:\n{context}\n\nQuestions:\n{questions}\nAnswers:\n{answers}",
             "suffix": "\nAnswers:\n"

             },
}