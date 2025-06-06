from util import log_flush
from data_loader.loader_util import get_dataloader
from teacher_model import TeacherModel
from student_model import StudentModel
from config import experience_constructor_meta
from experiment_analyzer.retrospect_analyzer import RetrospectAnalyzer
import datetime
import random


class RetroTester:
    def __init__(self, id: str, subject: str, dataset: str, subset:str, test_amount: int, round_amount: int, model, analyzer: RetrospectAnalyzer, trace_scores: bool, use_experience: bool, save_experience: bool, save_distill_experience: bool = True):
        self.id = id
        self.dataset = dataset
        self.use_experience = use_experience
        self.save_experience = save_experience
        self.save_distill_experience = save_distill_experience
        if not self.use_experience:
            assert not self.save_experience, "If not using experience, shall not save experience."

        # logger setting
        self.fileIO = open(f"./log/retrotester_{self.id}_{self.dataset}.log", 'a')
        log_flush(self.fileIO, "===============================")
        log_flush(self.fileIO, f"Initialize retro tester id({self.id}) on dataset({self.dataset}) " + str(datetime.datetime.now()))

        # load data
        self.dataloader = get_dataloader(dataset, subset)
        self.test_amount = test_amount
        if self.test_amount > self.dataloader.get_size():
              self.test_amount = self.dataloader.get_size()
              print(f"Tester")
        self.dataset = self.dataloader.load_n_from_front(self.test_amount)
        self.round_amount = round_amount

        # set up analyzer
        self.analyzer = analyzer

        # set up teacher and student model
        self.trace_scores = trace_scores
        self.teacher_model = TeacherModel(subject=subject, model=model, trace_score=trace_scores)
        self.student_model = StudentModel(model=model)

    def start(self, n: int, is_random: bool = False):
        """
        n: number of testing in this round
        is_random: whether to random test
        """
        log_flush(self.fileIO, f"Start testing with test set with size({self.test_amount}) at time ({str(datetime.datetime.now())}) ...")
        for i in range(n):
            j = i
            if j >= self.test_amount:
                j -= self.test_amount
            if is_random:
                j = random.randint(0, self.test_amount - 1)
            log_flush(self.fileIO, "===============================")
            log_flush(self.fileIO, f"+++ on test datapoint {i + 1}/{n} on question {j} ...")
            print(f"+++ on test datapoint {i + 1}/{n} on question {j} ...")
            cur_question = self.dataset[j]
            log_flush(self.fileIO, f"question: ({cur_question["question"]}), truth: ({cur_question["truth"]}), options: ({cur_question["options"]})")
            # set up teacher and student model for the question
            self.teacher_model.load_question(cur_question)
            self.analyzer.load_question(j, self.teacher_model.question, self.teacher_model.clean_truth, self.round_amount)
            # start the rounds
            accurate_count = 0
            contexts = []
            if self.use_experience:
                if self.trace_scores:
                    scores, contexts = self.teacher_model.get_question_related_experience_context()
                else:
                    scores = []
                    contexts = self.teacher_model.get_question_related_experience_context()
            log_flush(self.fileIO, f"contexts: {contexts}")
            for _ in range(self.round_amount):
                # join the context into string
                context = " --- ".join(contexts)
                student_answer = self.student_model.solve_question(self.teacher_model.question, self.teacher_model.plain_options, context)
                is_correct_answer = self.teacher_model.exam_result(student_answer)
                if is_correct_answer:
                    log_flush(self.fileIO, f"Student model answer is CORRECT: {student_answer}.")
                    accurate_count += 1
                else:
                    log_flush(self.fileIO, f"Student model answer is WRONG: {student_answer} ")
                # save the experience
            if self.save_experience:
                self.teacher_model.save_conversation(self.save_distill_experience)
            # write to analyzer
            if self.use_experience:
                log_flush(self.fileIO, f"Test on datapoint {i + 1}/{n} on question {j}/ finished. Experience accurate count: {accurate_count}/{self.round_amount}.")
                self.analyzer.load_experience_result(j, accurate_count)
            else:
                log_flush(self.fileIO, f"Test on datapoint {i + 1}/{n} on question {j}/ finished. Plain accurate count: {accurate_count}/{self.round_amount}.")
                self.analyzer.load_plain_result(j, accurate_count)
        self.analyzer.save_to_csv()
