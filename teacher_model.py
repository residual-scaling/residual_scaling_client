import threading
import time
from model.model_loader import load_model
from rsdb_connector import RSDBConnector
from config import server_setting
from util import log_flush
import datetime


class TeacherModel:
    def __init__(self, model_path, model_type: str = "local"):
        """
        TeacherModel class to handle the loading of the model and the connectors for the residual scaling database.
        """
        self.model_type = model_type
        self.model = load_model(model_type, model_path)
        
        # logger setting
        self.fileIO = open('./log/distill_log.txt', 'a')
        log_flush(self.fileIO, "===============================")
        log_flush(self.fileIO, "Start teacher model on " + str(datetime.datetime.now()))

        # question setting
        self.question = None
        self.resoning_process = None
        self.answer = None
        self.options = []
        self.conversation = ""
        self.try_count = 0

    def load_question(self, question: str, reasoning_process: str = None, answer: str = None, options: list = []):
        """
        Load the question into the model.
        """
        log_flush(self.fileIO, "+++++++++++++++++++++++++++++++")
        log_flush(self.fileIO, "Question loaded: " + question)
        log_flush(self.fileIO, "Reasoning process loaded: " + str(reasoning_process))
        log_flush(self.fileIO, "Answer loaded: " + str(answer))
        log_flush(self.fileIO, "Options loaded: " + str(options))
        self.question = question
        self.resoning_process = reasoning_process
        self.answer = answer
        self.options = options
        self.try_count = 0
        log_flush(self.fileIO, "Conversation cleared and reloated.")
        self.conversation = ""
        self.conversation += f"Question: '{question}', "
        if len(self.options) > 0:
            self.conversation += f"Options:  '{str(self.options)}', "

    def exam_result(self, result: str) -> bool:
        """
        Save the exam result to the conversation.
        """
        log_flush(self.fileIO, "Exam result loaded: " + result)
        self.conversation += f"Result: '{result}', "
        
    def get_full_conversation(self):
        """
        Get the full conversation.
        """
        return self.conversation

        

