import pandas as pd


class RetrospectAnalyzer:
    def __init__(self, row_len: int):
        self.row_len = row_len
        self.df = pd.DataFrame(index=[i for i in range(self.row_len)], 
                        columns=[
                            "question_body",
                            "truth",
                            "round_num",
                            "exp_correctness_record",
                            "plain_correctness_record",
                        ])
        self.df["plain_correctness_record"] = [[] for _ in range(self.row_len)]
        self.df["exp_correctness_record"] = [[] for _ in range(self.row_len)]

    def load_question(self, index: int, question_body: str, truth: str, round_num: int):
        self.df.at[index, "question_body"] = question_body
        self.df.at[index, "truth"] = truth
        self.df.at[index, "round_num"] = round_num

    def load_plain_result(self, index:int, record: int):
        self.df.at[index, "plain_correctness_record"].append(record)

    def load_experience_result(self, index:int, record: int):
        self.df.at[index, "exp_correctness_record"].append(record)

    def save_to_csv(self):
        self.df.to_csv("./log/retrospect_experiment_anlysis.csv", encoding='utf-8')
