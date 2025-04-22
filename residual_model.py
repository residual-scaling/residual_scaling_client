import threading
import time
from model.model_loader import load_model
from rsdb_connector import RSDBConnector
from config import server_setting


def load_connectors():
    connectors = {}
    assert len(server_setting["host"]) == len(server_setting["port"]) == len(server_setting["field"]), "ERROR: server_setting length mismatch!"
    for i in range(len(server_setting["host"])):
        connectors[server_setting["field"][i]] = RSDBConnector(server_setting["host"][i], server_setting["port"][i])
    return connectors


class ResidualModel:
    def __init__(self, model, model_type: str = "local", db_type: str = "vector"):
        self.model_type = model_type
        self.model = load_model(model_type, model)
        self.server_fields = server_setting["field"]
        self.connectors = load_connectors()
        self.distillation_queue = []
        self.queue_lock = threading.Lock()
        
        self.distillation_thread = threading.Thread(target=self.process_distillation_queue, daemon=True)
        self.distillation_thread.start()
    
    def process_distillation_queue(self):
        log_file = open('./log/distill_log.txt', 'a')
        log_file.write("[Distill] Start distillation thread\n")
        print("[Distill] Start distillation thread")
        log_file.flush()

        while True:
            time.sleep(3)
            log_file.write("[Distill] check distillation queue\n")
            print("[Distill] check distillation queue")
            while len(self.distillation_queue) > 0:
                with self.queue_lock:
                    q_field, cur_context = self.distillation_queue.pop(0)
                    log_file.write(f"[Distill] distilling 1 out of {len(self.distillation_queue) + 1}\n")
                    print(f"[Distill] distilling 1 out of {len(self.distillation_queue) + 1}")
                distilled_context = self.model.generate_distillation(cur_context)
                # sleep 0.1s to give user query processing the priority, creating a pseudo priority lock
                time.sleep(0.1)
                self.connectors[q_field].put(distilled_context)
            log_file.write("[Distill] distillation queue clean!\n")
            print("[Distill] distillation queue clean!")
            # write to disk, force the result to appear immidiately in the file
            log_file.flush()

    def query_without_residual(self, query: str) -> str:
        print("### Start query_without_residual ### ")
        result = self.model.generate_answer(query)
        print("### Finish query_without_residual ### ")
        return result

    def query_with_residual(self, query: str, question_type: str) -> str:
        print("### Start query_with_residual() ### ")
        related_contexts = self.connector[question_type].get(query)
        with self.queue_lock:
            print(f"--- related_contexts len: {len(related_contexts)}---", )
            context = ",".join(related_contexts)
            result = self.model.generate_answer(query, context)
            self.distillation_queue.append((question_type, f"Query: {query}; Result: {result}"))
            print("### Finish query_with_residual() ### ")
        return result

    def query(self, question: str) -> str:
        print("### Start query() ### ")
        question_type = self.model.get_question_type(question)
        if question_type in self.server_fields:
            print(f"Query type: {question_type}, use residual scaling")
            result = self.query_with_residual(question, question_type)
        else:
            print(f"Query type: {question_type}, DO NOT use residual scaling")
            result = self.query_without_residual(question)
        print("### Finish query() ### ")
        return result
