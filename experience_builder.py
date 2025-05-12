from util import log_flush
from data_loader.loader_util import get_dataloader
from teacher_model import TeacherModel
from student_model import StudentModel
from config import experience_constructor_meta
import datetime


class ExperienceBuilder:
    def __init__(self, id: str, subject: str, dataset: str, subset:str, train_percent: float, model, save_distill_experience: bool = True, save_false_experience: bool = False):
        self.id = id
        self.dataset = dataset
        self.max_round = experience_constructor_meta["max_round"]
        self.save_distill_experience = save_distill_experience
        self.save_false_experience = save_false_experience
        self.train_percent = train_percent

        # set dataloader
        self.dataloader = get_dataloader(dataset, subset)
        # set train and test set
        self.train_set, self.test_set = self.dataloader.load_train_test_set(train_percent, random_shuffle=False)

        # set up teacher and student model
        self.teacher_model = TeacherModel(subject=subject, model=model)
        self.student_model = StudentModel(model=model)

        # logger setting
        self.fileIO = open(f"./log/ExperienceBuilder_{self.id}_{self.dataset}.log", 'a')
        log_flush(self.fileIO, "===============================")
        log_flush(self.fileIO, f"Initialize Experience Builder id({self.id}) on dataset({self.dataset}) " + str(datetime.datetime.now()))

    def start(self):
        train_size = len(self.train_set)
        log_flush(self.fileIO, f"Start Experience Building ont train set with size({train_size}) ...")
        # loop though the train set
        for i in range(train_size):
            log_flush(self.fileIO, f"+++ on train datapoint {i + 1}/{train_size} ...")
            log_flush(self.fileIO, f"question: ({self.train_set[i]["question"]}), truth: ({self.train_set[i]["truth"]}), options: ({self.train_set[i]["options"]})")
            print(f"+++ on train datapoint {i + 1}/{train_size} ...")
            self.teacher_model.load_question(self.train_set[i])
            # get correct answer with max try
            is_correct_answer = False
            for j in range(self.max_round):
                log_flush(self.fileIO, f"try {j + 1}/{self.max_round} ...")
                student_answer = self.student_model.solve_question(self.teacher_model.question, self.teacher_model.options, self.teacher_model.get_context())
                is_correct_answer = self.teacher_model.exam_result(student_answer)
                if is_correct_answer:
                    log_flush(self.fileIO, f"Student model answer is CORRECT: {student_answer} on try {j + 1}.")
                    break
                else:
                    log_flush(self.fileIO, f"Student model answer is WRONG: {student_answer} on try {j + 1}, try again.")
            if not is_correct_answer and not self.save_false_experience:
                log_flush(self.fileIO, f"Do not save wrong answer experience.")
                continue
            # save the experience
            self.teacher_model.save_conversation(self.save_distill_experience)

    def save_experience_db(self):
        self.teacher_model.save_experience_db()
