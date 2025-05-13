import threading
import time
from model.model_loader import load_model
from rsdb_connector import RSDBConnector
from config import server_setting
from util import log_flush
import datetime


class TeacherModel:
    def __init__(self, subject: str, model, trace_score: bool):
        """
        Teacher model save question to ask and save the conversation(experience).
        """
        self.model = model

        # logger setting
        self.fileIO = open('./log/teacher_log.txt', 'a')
        log_flush(self.fileIO, "===============================")
        log_flush(self.fileIO, "Start teacher model on " + str(datetime.datetime.now()))

        # set subject and connect to the server of the subject
        if subject not in server_setting["field"]:
            raise ValueError(f"ERROR: In TeacherModel.__init__(), {subject} is not in the server setting!")
        self.subject = subject
        field_ind = server_setting["field"].index(subject)
        self.trace_score = trace_score
        self.connector = RSDBConnector(server_setting["host"][field_ind], server_setting["port"][field_ind], self.trace_score)

        # fixed question setting
        self.question = None
        self.reasoning = None
        self.truth = None
        self.options = []

        # flex conversation setting
        self.wrong_answers = []
        self.correct_answers = []  # there should only be one
        # self.conversation = ""
        self.try_count = 0


    def reset_flex_conversation_setting(self):
        self.wrong_answers = []
        self.correct_answers = []
        # self.conversation = ""
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
        log_flush(self.fileIO, "Conversation cleared and reloaded.")

        # reset flex conversation setting
        self.reset_flex_conversation_setting()

    def exam_result(self, answer: str) -> bool:
        """
        Save the exam answer to the conversation.
        """
        self.try_count += 1
        if answer == self.truth:
            log_flush(self.fileIO, f"Answer ({answer}) is correct.")
            self.correct_answers.append(answer)
            if len(self.correct_answers) > 1:
                print("WARNING: in exam_result(), there are more than 1 correct answer")
                log_flush("WARNING: in exam_result(), there are more than 1 correct answer")
            # self.conversation += f" The result of the previous attempt is: '{answer}', and it is the CORRECT answer! "
            return True
        else:
            if answer not in self.wrong_answers:
                self.wrong_answers.append(answer)
            log_flush(self.fileIO, f"Truth is ({self.truth}), Answer is ({answer}), answer is wrong.")
            # self.conversation += f" The result of the previous attempt is: '{answer}', and it is a WRONG answer! "
        return False
        
    def get_full_conversation(self):
        """
        Get the full conversation.
        """
        full_conversation = f"Question: '{self.question}', "
        if len(self.correct_answers) > 0:
            full_conversation += f"CORRECT answer: '{self.correct_answers[0]}', "
        if len(self.wrong_answers) > 0:
            full_conversation += f"WRONG answer: '{self.wrong_answers}'. "
        return full_conversation
    
    def get_context(self) -> str:
        if len(self.correct_answers) > 0 or len(self.wrong_answers) > 0:
            return self.get_full_conversation()
        return ""

    def get_distilled_conversation(self):
        return self.model.generate_QA_distillation(self.question, self.correct_answers, self.wrong_answers)

    def save_conversation(self, save_distilled: bool = True):
        """
        Save the conversation to the database.
        """
        to_save_conversation = self.get_full_conversation()
        log_flush(self.fileIO, f"Full conversation: ({to_save_conversation})")
        if save_distilled:
            log_flush(self.fileIO, "Distilling")
            to_save_conversation = self.get_distilled_conversation()
        log_flush(self.fileIO, "Conversation to save: " + to_save_conversation)
        self.connector.put(to_save_conversation)

    def save_experience_db(self):
        """
        Tell the server to save backup database.
        """
        self.connector.save_db()

    def get_question_related_experience_context(self) -> list:
        """
        Get the question related contexts from the database.
        """
        if self.trace_score:
            # get the context with score
            scores, related_contexts = self.connector.get(" ".join(self.options))
            log_flush(self.fileIO, f"Retrieve from VectorDB, options: ({" ".join(self.options)}), related scores: ({scores}), related contexts: ({related_contexts})")
            # scores, related_contexts = self.connector.get(self.question)
            # log_flush(self.fileIO, f"Retrieve from VectorDB, original question: ({self.question}), related scores: ({scores}), related contexts: ({related_contexts})")
            return scores, related_contexts
        else:
            related_contexts = self.connector.get(self.question)
            log_flush(self.fileIO, f"Retrieve from VectorDB, original question: ({self.question}), only contexts: ({str(related_contexts)})")
            return related_contexts
