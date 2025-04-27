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
