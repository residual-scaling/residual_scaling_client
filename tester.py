from data_loader.loader_util import get_dataloader
from teacher_model import TeacherModel
from student_model import StudentModel
from util import log_flush
from experiment_analyzer.single_run_analyzer import SingleRunAnalyzer
import datetime


class Tester:
    def __init__(self, id: str, subject: str, dataset: str, subset:str, train_percent: float, model, analyzer: SingleRunAnalyzer, trace_scores: bool, use_experience: bool):
        self.id = id
        self.dataset = dataset
        self.train_percent = train_percent
        self.use_experience = use_experience

        # set dataloader
        self.dataloader = get_dataloader(dataset, subset)
        # set train and test set
        self.train_set, self.test_set = self.dataloader.load_train_test_set(train_percent, random_shuffle=False)

        # set up teacher and student model
        self.trace_scores = trace_scores
        self.teacher_model = TeacherModel(subject=subject, model=model, trace_score=trace_scores)
        self.student_model = StudentModel(model=model)

        # set up analyzer
        self.analyzer = analyzer

        # logger setting
        self.fileIO = open(f"./log/Tester_{self.id}_{self.dataset}.log", 'a')
        log_flush(self.fileIO, "===============================")
        log_flush(self.fileIO, f"Initialize Experience Builder id({self.id}) on dataset({self.dataset}) " + str(datetime.datetime.now()))

    def start(self):
        accurate_count = 0
        test_size = len(self.test_set)
        log_flush(self.fileIO, f"Start testing with test set with size({test_size}) ...")
        # loop though the test set
        for i in range(test_size):
            log_flush(self.fileIO, f"+++ on test datapoint {i + 1}/{test_size} ...")
            print(f"+++ on test datapoint {i + 1}/{test_size} ...")
            log_flush(self.fileIO, f"question: ({self.train_set[i]["question"]}), truth: ({self.train_set[i]["truth"]}), options: ({self.train_set[i]["options"]})")
            self.teacher_model.load_question(self.test_set[i])
            self.analyzer.load_question(i, self.teacher_model.question, self.teacher_model.truth)
            
            contexts = []
            if self.use_experience:
                if self.trace_scores:
                    scores, contexts = self.teacher_model.get_question_related_experience_context()
                else:
                    scores = []
                    contexts = self.teacher_model.get_question_related_experience_context()
            # join the context into string
            context = " --- ".join(contexts)
            student_answer = self.student_model.solve_question(self.teacher_model.question, self.teacher_model.options, context)
            is_correct_answer = self.teacher_model.exam_result(student_answer)
            if is_correct_answer:
                log_flush(self.fileIO, f"Student model answer is CORRECT: {student_answer}.")
                accurate_count += 1
            else:
                log_flush(self.fileIO, f"Student model answer is WRONG: {student_answer} ")
            # write to analyzer
            if self.use_experience:
                self.analyzer.load_experience_result(i, student_answer, is_correct_answer, scores, contexts)
            else:
                self.analyzer.load_plain_result(i, student_answer, is_correct_answer)
        self.analyzer.save_to_csv()
        log_flush(self.fileIO, f"Test finished. Accurate count: {accurate_count}/{test_size}.")
        return accurate_count, test_size