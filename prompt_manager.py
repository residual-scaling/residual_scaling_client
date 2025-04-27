from config import prompt_hyperparameter, server_setting
from util import cut_last_appearance, clean_string


class PromptManager:
    def __init__(self):
        self.max_summarize_len = prompt_hyperparameter["max_summarize_len"]
        self.fields = server_setting["field"]

    def generate_llama_summarize_prompt(self, txt: str) -> str:
        prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
        Summarize this text in {self.max_summarize_len - 1}-{self.max_summarize_len} concise sentences:

        {txt}

        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        Summary:"""
        return prompt

    def retrieve_llama_summarized_content(self, txt: str) -> str:
        keyword = "Summary:"
        return clean_string(cut_last_appearance(txt, keyword))

    def generate_llama_generation_prompt(self, query: str, context: list) -> str:
        if len(context) == 0:
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
                    You are a helpful AI assistant.<|eot_id|>
                    <|start_header_id|>user<|end_header_id|>
                    {query}<|eot_id|>
                    <|start_header_id|>assistant<|end_header_id|>"""
        else:
            # there exists some context
            concated_context = ",".join(context)
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
                    You are a helpful AI assistant. Always use the provided context to answer.<|eot_id|>
                    <|start_header_id|>user<|end_header_id|>
                    Context: "{concated_context}"

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

    def generate_llama_percise_question_prompt(self, query: str, options: list = [], context: list = []) -> str:
        # create appropriate header
        header = "You are a helpful AI assistant. The final answer should be as short and percise as possible."
        if len(options) > 0:
            header += " Choose the answer from the options."
        if len(context) > 0:
            header += " Always use the provided context to answer."
        header += " Answer the final answer in the end after #### "

        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
                {header}<|eot_id|>
                <|start_header_id|>user<|end_header_id|>\n"""
        
        # cosntruct the prompt
        if len(options) > 0:
            prompt += f"Options: '{str(options)}'\n"
        if len(context) > 0:
            concated_context = ",".join(context)
            prompt += f"Context: '{concated_context}'\n"
        prompt += f"""Request: {query}<|eot_id|>
                <|start_header_id|>assistant<|end_header_id|>"""
        return prompt
    
    def retrieve_llama_percise_answer(self, txt: str) -> str:
        keyword = "####"
        return clean_string(cut_last_appearance(txt, keyword)).strip()
