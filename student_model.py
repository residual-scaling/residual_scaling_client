from model.model_loader import load_model
from util import log_flush
import datetime
from prompt_manager import PromptManager


class StudentModel:
    def __init__(self, model):
        """
        Student model solve question, clean and return the data, that's it.
        """
        self.model = model

        # prompt manager
        self.pm = PromptManager()

        # logger setting
        self.fileIO = open('./log/student_log.txt', 'a')
        log_flush(self.fileIO, "===============================")
        log_flush(self.fileIO, "Start teacher model on " + str(datetime.datetime.now()))

    def solve_question(self, question: str, options: list, context: str) -> str:
        """
        Solve question with input conversation from the teacher model.
        """
        # gather question data
        log_flush(self.fileIO, "+++++++++++++++++++++++++++++++")
        log_flush(self.fileIO, f"Solve question: {question}, with options: {options} and previous conetxt: '{context}'")

        # get prompt
        prompt = self.pm.generate_llama_percise_question_prompt(question, options, context)
        # log_flush(self.fileIO, "Prompt: " + prompt)
        # generate result
        result = self.model.clean_pipe(prompt)
        log_flush(self.fileIO, "-------------")
        log_flush(self.fileIO, f"Result: {result}")
        # print("result: ", result)
        answer = self.pm.retrieve_llama_percise_answer(result)
        log_flush(self.fileIO, "Answer: " + answer)
        return answer
    
    def get_percise_answer_basic(self, question: str, options: list) -> str:
        """
        TODO duplicated, solve_question() with exmpty context act the same.
        Get precise answer with input conversation from the teacher model.
        """
        log_flush(self.fileIO, "+++++++++++++++++++++++++++++++")
        log_flush(self.fileIO, f"Get percise answer WITHOUT residual, question: {question}, with options: {options}")
        # get prompt
        prompt = self.pm.generate_llama_percise_question_prompt(question, options, "")
        log_flush(self.fileIO, "Prompt: " + prompt)
        # generate result
        result = self.model.clean_pipe(prompt)
        # print("result: ", result)
        answer = self.pm.retrieve_llama_percise_answer(result)
        log_flush(self.fileIO, "Answer: " + answer)
        return answer

    def get_option_confidence(self, question: str, target_option: str, options: list, context: str) -> (bool, str):
        """
        Solve question with input conversation from the teacher model.
        """
        # gather question data
        log_flush(self.fileIO, "+++++++++++++++++++++++++++++++")
        log_flush(self.fileIO, f"Get confidence for option: {target_option}, question: {question}, with options: {options} and previous conetxt: '{context}'")

        # get prompt
        prompt = self.pm.generate_llama_percise_confidence_prompt(question, target_option, options, context)
        # log_flush(self.fileIO, "Prompt: " + prompt)
        # generate result
        result = self.model.clean_pipe(prompt)
        log_flush(self.fileIO, "-------------")
        log_flush(self.fileIO, f"Result: {result}")
        # print("result: ", result)
        answer = self.pm.retrieve_llama_percise_confidence(result)
        return answer
