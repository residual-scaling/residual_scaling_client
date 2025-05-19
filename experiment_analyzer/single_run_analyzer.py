import pandas as pd


class SingleRunAnalyzer:
    def __init__(self, row_len: int):
        self.row_len = row_len
        self.df = pd.DataFrame(index=[i for i in range(self.row_len)], 
                        columns=[
                            "question_body",
                            "truth",
                            "plain_answers",
                            "exp_answers",
                            "plain_correctness",
                            "exp_correctness",
                            "avg_bm25score",
                            "bm25scores",
                            "contexts",
                        ])
        self.df["plain_answers"] = [[] for _ in range(self.row_len)]
        self.df["exp_answers"] = [[] for _ in range(self.row_len)]
        self.df["plain_correctness"] = [0 for _ in range(self.row_len)]
        self.df["exp_correctness"] = [0 for _ in range(self.row_len)]
        self.df["bm25scores"] = [[] for _ in range(self.row_len)]

    def load_question(self, index: int, question_body: str, truth: str):
        self.df.at[index, "question_body"] = question_body
        self.df.at[index, "truth"] = truth

    def load_plain_result(self, index:int, answer: str, is_correct: bool):
        self.df.at[index, "plain_answers"].append(answer)
        if is_correct:
            self.df.at[index, "plain_correctness"] += 1

    def load_experience_result(self, index:int, answer: str, is_correct: bool, scores: list, contexts: list):
        self.df.at[index, "exp_answers"].append(answer)
        if is_correct:
            self.df.at[index, "exp_correctness"] += 1
        self.df.at[index, "bm25scores"] = scores
        # -1 indicates no scores
        self.df.at[index, "avg_bm25score"] = sum(scores) / len(scores) if len(scores) > 0 else -1
        self.df.at[index, "contexts"] = contexts
        

    def save_to_csv(self):
        self.df.to_csv("./log/single_run_experiment_anlysis.csv", encoding='utf-8')
