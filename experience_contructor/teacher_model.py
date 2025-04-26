import threading
import time
from model.model_loader import load_model
from rsdb_connector import RSDBConnector
from config import server_setting
from util import log_flush


class TeacherModel:
    def __init__(self, model, model_type: str = "local", db_type: str = "vector"):
        """
        TeacherModel class to handle the loading of the model and the connectors for the residual scaling database.
        """
        self.model_type = model_type
        self.model = load_model(model_type, model)
        self.server_fields = server_setting["field"]
        self.connectors = load_connectors()
        self.distillation_queue = []
        self.queue_lock = threading.Lock()
        
        self.distillation_thread = threading.Thread(target=self.process_distillation_queue, daemon=True)
        self.distillation_thread.start()
        log_file = open('./log/distill_log.txt', 'a')
