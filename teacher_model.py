import threading
import time
from model.model_loader import load_model
from rsdb_connector import RSDBConnector
from config import server_setting
from util import log_flush
import datetime


class TeacherModel:
    def __init__(self):
        """
        Teacher model save question to ask and save the conversation(experience).
        """
        # self.model_type = model_type
        # self.model = load_model(model_type, model_path)
        
        # logger setting
        self.fileIO = open('./log/teacher_log.txt', 'a')
        log_flush(self.fileIO, "===============================")
        log_flush(self.fileIO, "Start teacher model on " + str(datetime.datetime.now()))

        # question setting
        self.question = None
        self.reasoning = None
        self.truth = None
        self.options = []
        self.conversation = ""
        self.try_count = 0

        # rsdb setting
        self.rsdb

    def load_question(self, question_data: dict):
        """
        Load the question into the model.
        """
        self.question = question_data["question"]
        self.reasoning = question_data["reasoning"]
        self.truth = question_data["truth"]
        self.options = question_data["options"]
        log_flush(self.fileIO, "+++++++++++++++++++++++++++++++")
        log_flush(self.fileIO, "Question loaded: " + self.question)
        log_flush(self.fileIO, "Reasoning loaded: " + self.reasoning)
        log_flush(self.fileIO, "Truth loaded: " + self.truth)
        log_flush(self.fileIO, "Options loaded: " + str(self.options))
        self.try_count = 0
        log_flush(self.fileIO, "Conversation cleared and reloaded.")
        self.conversation = ""

    def exam_result(self, answer: str) -> bool:
        """
        Save the exam answer to the conversation.
        """
        self.try_count += 1
        if answer == self.truth:
            log_flush(self.fileIO, f"Answer ({answer}) is correct.")
            self.conversation += f" The result of the previous attempt is: '{answer}', and it is the CORRECT answer! "
            return True
        else:
            log_flush(self.fileIO, f"Truth is ({self.truth}), Answer is ({answer}), answer is wrong.")
            self.conversation += f" The result of the previous attempt is: '{answer}', and it is a WRONG answer! "
        return False

        
    def get_full_conversation(self):
        """
        Get the full conversation.
        """
        full_conversation = f"Question: '{self.question}', "
        if len(self.options) > 0:
            full_conversation += f"Options of answers: {str(self.options)}, "
        return full_conversation + self.conversation
    
    def get_new_question_state(self):
        """
        Get the new question state.
        """
        return {
            "question": self.question,
            "options": self.options,
            "conversation": self.conversation
        }

        

