from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from model.base_model import BaseModel
from prompt_manager import PromptManager
from config import model_hyperparameter
from util import clean_string


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

    def generate_distillation(self, txt: str) -> str:
        print("[Local Model] Distilling...")
        summerization_prompt = self.pm.generate_llama_summarize_prompt(txt)
        # print("summerization_prompt: ", summerization_prompt)
        summarized_result = self.pipe(summerization_prompt)[0]['generated_text']
        # print("summarized_result: ", summarized_result)
        summarized_txt = clean_string(self.pm.retrieve_llama_summarized_content(summarized_result))
        # print("summarized_txt: ", summarized_txt)
        return summarized_txt

    def generate_answer(self, txt: str, context: list) -> str:
        print("[Local Model] Generating answer...")
        generation_prompt = self.pm.generate_llama_generation_prompt(txt, context)
        # print("generation_prompt: ", generation_prompt)
        generation_result = self.pipe(generation_prompt)[0]['generated_text']
        # print("generation_result: ", generation_result)
        answer = clean_string(self.pm.retrieve_llama_generation_content(generation_result))
        # print("answer: ", answer)
        return answer

