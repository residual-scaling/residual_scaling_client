from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from model.base_model import BaseModel
from prompt_manager import PromptManager
from config import model_hyperparameter
from util import clean_string, get_generated_text


class LocalModel(BaseModel):
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype="auto"
        )
        self.pm = PromptManager()
        self.temperature = model_hyperparameter["temperature"]
        self.pipe = pipeline("text-generation", 
                            model=self.model, 
                            tokenizer=self.tokenizer,
                            do_sample=True,
                            temperature=self.temperature
                            )
        
    def clean_pipe(self, prompt: str) -> str:
        result = get_generated_text(self.pipe(prompt))
        return result

    def generate_distillation(self, txt: str) -> str:
        print("[Local Model] Distilling...")
        summerization_prompt = self.pm.generate_llama_summarize_prompt(txt)
        # print("summerization_prompt: ", summerization_prompt)
        summarized_result = self.clean_pipe(summerization_prompt)
        # print("summarized_result: ", summarized_result)
        summarized_txt = self.pm.retrieve_llama_summarized_content(summarized_result)
        # print("summarized_txt: ", summarized_txt)
        return summarized_txt

    def generate_answer(self, txt: str, context: str) -> str:
        print("[Local Model] Generating answer...")
        generation_prompt = self.pm.generate_llama_generation_prompt(txt, context)
        # print("generation_prompt: ", generation_prompt)
        generation_result = self.clean_pipe(generation_prompt)
        # print("generation_result: ", generation_result)
        answer = self.pm.retrieve_llama_generation_content(generation_result)
        # print("answer: ", answer)
        return answer

    def get_question_type(self, txt: str) -> str:
        print("[Local Model] Classifying question type...")
        classify_type_prompt = self.pm.generate_llama_classify_question_type(txt)
        # print("classify_type_prompt: ", classify_type_prompt)
        classify_type_result = self.clean_pipe(classify_type_prompt)
        # print("classify_type_result: ", classify_type_result)
        result_type = self.pm.retrieve_llama_generation_content(classify_type_result)
        # print("result_type: ", result_type)
        return result_type

