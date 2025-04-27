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
        # self.model_type = model_type
        # self.model = load_model(model_type, model_path)
        
        # logger setting
        self.fileIO = open('./log/distill_log.txt', 'a')
        log_flush(self.fileIO, "===============================")
        log_flush(self.fileIO, "Start teacher model on " + str(datetime.datetime.now()))

        # question setting
        self.question = None
        self.reasoning = None
        self.truth = None
        self.options = []
        self.conversation = ""
        self.try_count = 0

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
        log_flush(self.fileIO, "Conversation cleared and reloated.")
        self.conversation = ""
        self.conversation += f"Question: '{self.question}', "
        if len(self.options) > 0:
            self.conversation += f"Options: {str(self.options)}, "

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

        

