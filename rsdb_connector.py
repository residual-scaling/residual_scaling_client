import threading
import socket
from rsprotocol.protocol import RSMetaMsg, RSGetType, RSMetaType, RSMsgDelimiter


class RSDBConnector:
    def __init__(self, host, port, getwith_score=False):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((host, port))

        if getwith_score:
            self.get_type = RSGetType.WITH_SCORE
        else:
            self.get_type = RSGetType.WITHOUT_SCORE

        self.socket_lock = threading.Lock()

    def put(self, data: str):
        bs = data.encode('utf-8')
        with self.socket_lock:
            self.client_socket.sendall(RSMetaMsg(RSMetaType.CLIENT_PUT, RSGetType.NOT_RELATED, len(bs)).serialize())
            self.client_socket.sendall(bs)

    def handle_get_result(self, data: str):
        # split the data by the delimiter
        data = data.split(RSMsgDelimiter.RSJOINBLOCK)
        # if return only a list with an empty string, this mean db has no qualified result
        if len(data) == 1 and data[0] == '':
            return [], []
        
        if self.get_type == RSGetType.WITH_SCORE:
            # the first element is the score
            scores = data[0].split(RSMsgDelimiter.RS_SCORE_SEPERATOR)
            scores = [float(s) for s in scores]
            # the rest are the results
            results = data[1:]
        else:
            # all elements are the results
            scores = []
            results = data
        if self.get_type == RSGetType.WITH_SCORE and  len(results) != len(scores):
            raise ValueError(f"ERROR, In RSDBConnector.handle_get_result() The length of results ({results}) and scores ({scores}) are not equal")
        return scores, results

    def get(self, query: str) -> list:
        bs = query.encode('utf-8')
        with self.socket_lock:
            # send to server
            self.client_socket.sendall(RSMetaMsg(RSMetaType.CLIENT_GET, self.get_type, len(bs)).serialize())
            self.client_socket.sendall(bs)
            # receive from server
            return_meta = self.client_socket.recv(RSMetaMsg.get_size())
            return_meta = RSMetaMsg.deserialize(return_meta)
            data = self.client_socket.recv(return_meta.meta_len).decode('utf-8')
            scores, results = self.handle_get_result(data)
            # if len(results) > 0 and results[-1] == '':
            #     raise ValueError(f"The last element of result is empty string, results: {results}")
            if self.get_type == RSGetType.WITHOUT_SCORE:
                return results
            return scores, results
    
    def save_db(self):
        with self.socket_lock:
            # send to server
            self.client_socket.sendall(RSMetaMsg(RSMetaType.CLIENT_SAVE, RSGetType.NOT_RELATED, 0).serialize())
            # TODO: check the return value
        