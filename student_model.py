from model.model_loader import load_model
from util import log_flush
import datetime
from prompt_manager import PromptManager


class StudentModel:
    def __init__(self, model_path: str, model_type: str = "local"):
        """
        Student model solve question, clean and return the data, that's it.
        """
        self.model_type = model_type
        self.model = load_model(model_type, model_path)

        # prompt manager
        self.pm = PromptManager()

        # logger setting
        self.fileIO = open('./log/student_log.txt', 'a')
        log_flush(self.fileIO, "===============================")
        log_flush(self.fileIO, "Start teacher model on " + str(datetime.datetime.now()))

    def solve_question(self, question_data) -> str:
        # gather question data
        question = question_data["question"]
        options = question_data["options"]
        conversation = question_data["conversation"]
        log_flush(self.fileIO, "+++++++++++++++++++++++++++++++")
        log_flush(self.fileIO, f"Solve question: {question}, with options: {options} and previous conversation: '{conversation}'")
        if conversation is not None and len(conversation) != 0:
            context = [conversation]
        else:
            context = []

        # get prompt
        prompt = self.pm.generate_llama_percise_question_prompt(question, options, context)
        log_flush(self.fileIO, "Prompt: " + prompt)
        # generate result
        result = self.model.clean_pipe(prompt)
        print("result: ", result)
        answer = self.pm.retrieve_llama_percise_answer(result)
        log_flush(self.fileIO, "Answer: " + answer)
        return answer
