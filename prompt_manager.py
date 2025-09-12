from config import prompt_hyperparameter, server_setting
from util import cut_last_appearance, cut_last_appearance_beforewards, clean_string


class PromptManager:
    def __init__(self):
        self.max_summarize_len = prompt_hyperparameter["max_summarize_len"]
        self.fields = server_setting["field"]

    def generate_llama_summarize_prompt(self, txt: str) -> str:
        prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
        Summarize this text in less than {self.max_summarize_len} concise sentences:

        {txt}

        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        Summary:"""
        return prompt

    def retrieve_llama_summarized_content(self, txt: str) -> str:
        keyword = "Summary:"
        return clean_string(cut_last_appearance(txt, keyword))

    def generate_llama_generation_prompt(self, query: str, context: str) -> str:
        if len(context) == 0:
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
                    You are a helpful AI assistant.<|eot_id|>
                    <|start_header_id|>user<|end_header_id|>
                    {query}<|eot_id|>
                    <|start_header_id|>assistant<|end_header_id|>"""
        else:
            # there exists some context
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
                    You are a helpful AI assistant. Always use the provided context to answer.<|eot_id|>
                    <|start_header_id|>user<|end_header_id|>
                    Context: "{context}"

                    Request: {query}<|eot_id|>
                    <|start_header_id|>assistant<|end_header_id|>"""
        return prompt
    
    def retrieve_llama_generation_content(self, txt: str) -> str:
        keyword = "<|end_header_id|>"
        return clean_string(cut_last_appearance(txt, keyword))

    def generate_llama_classify_question_type(self, txt: str) -> str:
        fields = ", ".join(f'"{item}"' for item in  self.fields)
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
                    Classify this question into exactly one category: {fields} or "others". Respond with only the category name as one word.<|eot_id|>
                    <|start_header_id|>user<|end_header_id|>
                    Question: {txt}<|eot_id|>
                    <|start_header_id|>assistant<|end_header_id|>"""
        return prompt

    def retrieve_llama_classify_question_type(self, txt: str) -> str:
        keyword = "<|start_header_id|>assistant<|end_header_id|>"
        result_type = clean_string(cut_last_appearance(txt, keyword))
        if result_type not in [self.fields, "others"]:
            raise ValueError(f"ERROR: in retrieve_llama_classify_question_type() Invalid type: {result_type}. Expected one of {self.fields} or 'others'.")
        return result_type

    def generate_llama_percise_question_prompt(self, query: str, options: list, context: str) -> str:
        # create appropriate header
        header = "You are a helpful AI assistant. The final answer should be as short and percise as possible."
        if len(options) > 0:
            header += " Choose the answer from the options."
        if len(context) > 0:
            header += " Use the provided context to answer."
        header += " Answer the final answer in the end after #### "

        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
                {header}<|eot_id|>
                <|start_header_id|>user<|end_header_id|>\n"""
        
        # cosntruct the prompt
        if len(options) > 0:
            prompt += f"Options: '{str(options)}'\n\n"
        if len(context) > 0:
            prompt += f"Context: '[{context}]'\n\n"
        prompt += f"""Request: {query}<|eot_id|>
                <|start_header_id|>assistant<|end_header_id|>"""
        return prompt
    
    def retrieve_llama_percise_answer(self, txt: str) -> str:
        keyword = "####"
        return clean_string(cut_last_appearance(txt, keyword)).strip()

    def generate_llama_MC_distillation_prompt(self, question: str, correct_answers: list, wrong_answers: list) -> str:
        correct_answer = ",".join(correct_answers)
        wrong_answer = ",".join(wrong_answers)
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            Summary this input into one sentence containing: 
            1. The exact question subject 
            2. Correct answer (if provided)
            3. Wrong answers (if provided)
            Use natural language, not in a question and answer way. Do not question the correctness of the answers. Use the given information and do not add any information beyond what is given.<|eot_id|>
            <|start_header_id|>user<|end_header_id|>
            Input:
            Question: "{question}"
            Correct answer: "{correct_answer}"
            Wrong answer: "{wrong_answer}"<|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>
            Summary:"""
        return prompt
    
    def retrieve_llama_general_reponse(self, txt: str) -> str:
        keyword = "<|start_header_id|>assistant<|end_header_id|>"
        return clean_string(cut_last_appearance(txt, keyword)).strip()

    def generate_llama_percise_confidence_prompt(self, query: str, target_option: str, options: list, context: str) -> str:
        # create appropriate header
        header = "You are a helpful AI assistant. The final answer should be as short and percise as possible."
        if len(options) > 0:
            header += " Choose the answer from the options."
        if len(context) > 0:
            header += " Use the provided context to answer."
        header += " Answer the final answer in the end after #### "

        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
                {header}<|eot_id|>
                <|start_header_id|>user<|end_header_id|>\n"""
        
        # cosntruct the prompt
        if len(options) > 0:
            prompt += f"Options: '{str(options)}'\n\n"
        if len(context) > 0:
            prompt += f"Context: '[{context}]'\n\n"
        prompt += f"""Request: What is your confidence that '{target_option}' is correct answer to question: {query}<|eot_id|>
                <|start_header_id|>assistant<|end_header_id|>"""
        return prompt
    
    def retrieve_llama_percise_confidence(self, txt: str) -> (bool, str):
        keyword = "####"
        result = cut_last_appearance(txt, keyword).strip()
        if "%" in result:
            result = cut_last_appearance_beforewards(result, "%").strip()
        if "\n" in result:
            result = cut_last_appearance_beforewards(result, "\n").strip()
        result = clean_string(result).strip()
        print(f"Final result: {result}")
        if result.isnumeric():
            confidence = int(result)
            if confidence < 0 or confidence > 100:
                print(f"ERROR: in retrieve_llama_percise_confidence() Invalid confidence: {confidence}. Expected in range [0, 100].")
                return False, confidence
            return True, confidence
        else:
            return False, result
