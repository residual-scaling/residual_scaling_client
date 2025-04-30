import threading
import time
from model.model_loader import load_model
from rsdb_connector import RSDBConnector
from config import server_setting
from util import log_flush
import datetime


class TeacherModel:
    def __init__(self, subject: str, model_type: str = "local", model_path: str = None):
        """
        Teacher model save question to ask and save the conversation(experience).
        """
        self.model_type = model_type
        self.model = load_model(model_type, model_path)
        
        # logger setting
        self.fileIO = open('./log/teacher_log.txt', 'a')
        log_flush(self.fileIO, "===============================")
        log_flush(self.fileIO, "Start teacher model on " + str(datetime.datetime.now()))

        # set subject and connect to the server of the subject
        if subject not in server_setting["field"]:
            raise ValueError(f"ERROR: In TeacherModel.__init__(), {subject} is not in the server setting!")
        self.subject = subject
        field_ind = server_setting["field"].index(subject)
        self.connector = RSDBConnector(server_setting["host"][field_ind], server_setting["port"][field_ind])

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
    
    def get_distilled_conversation(self):
        full_conversation = self.get_full_conversation()
        return self.model.generate_distillation(full_conversation)
    
    def get_conversation_question_state(self):
        """
        Get the new question state.
        """
        return {
            "question": self.question,
            "options": self.options,
            "context": [self.conversation]
        }

    def get_plain_question_state(self):
        """
        Get the plain question state.
        """
        return {
            "question": self.question,
            "options": self.options,
            "context": []
        }
    
    def get_experience_question_state(self) :
        return {
            "question": self.question,
            "options": self.options,
            "context": self.get_question_related_contexts(),
        }
    
    def save_conversation(self, save_distilled: bool = True):
        """
        Save the conversation to the database.
        """
        to_save_conversation = self.get_full_conversation()
        if save_distilled:
            to_save_conversation = self.get_distilled_conversation()
        log_flush(self.fileIO, "Conversation to save: " + to_save_conversation)
        self.connector.put(to_save_conversation)

    def save_experience_db(self):
        """
        Tell the server to save backup database.
        """
        self.connector.save_db()

    def get_question_related_contexts(self) -> list:
        """
        Get the question related contexts from the database.
        """
        log_flush(self.fileIO, "Get question related contexts from the database.")
        related_contexts = self.connector.get(self.question)
        log_flush(self.fileIO, f"Related contexts: {str(related_contexts)}")
        return related_contexts
