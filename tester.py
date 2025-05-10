from data_loader.loader_util import get_dataloader
from teacher_model import TeacherModel
from student_model import StudentModel
from util import log_flush
from model.model_loader import load_model
import datetime


class Tester:
    def __init__(self, id: str, subject: str, dataset: str, subset: str, train_percent: float, model_type: str, model_path: str, use_experience: bool):
        self.id = id
        self.dataset = dataset
        self.train_percent = train_percent
        self.use_experience = use_experience

        # set dataloader
        self.dataloader = get_dataloader(dataset, subset)
        # set train and test set
        self.train_set, self.test_set = self.dataloader.load_train_test_set(train_percent, random_shuffle=False)

        # set up teacher and student model
        self.model = load_model(model_type, model_path)
        self.teacher_model = TeacherModel(subject=subject, model=self.model)
        self.student_model = StudentModel(model=self.model)

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
            self.teacher_model.load_question(self.test_set[i])
            
            if self.use_experience:
                log_flush(self.fileIO, f"Use experience to test.")
                question_state = self.teacher_model.get_experience_question_state()
                log_flush(self.fileIO, f"Experience question state: {str(question_state)}")
            else:
                question_state = self.teacher_model.get_plain_question_state()
            student_answer = self.student_model.solve_question(question_state)
            is_correct_answer = self.teacher_model.exam_result(student_answer)
            if is_correct_answer:
                log_flush(self.fileIO, f"Student model answer is CORRECT: {student_answer}.")
                accurate_count += 1
            else:
                log_flush(self.fileIO, f"Student model answer is WRONG: {student_answer} ")
        log_flush(self.fileIO, f"Test finished. Accurate count: {accurate_count}/{test_size}.")
        return accurate_count, test_size