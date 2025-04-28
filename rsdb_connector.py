import threading
import socket
from rsprotocol.protocol import RSMetaMsg, RSMetaType, RSMsgDelimiter


class RSDBConnector:
    def __init__(self, host, port):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((host, port))
        self.socket_lock = threading.Lock()

    def put(self, data: str):
        bs = data.encode('utf-8')
        with self.socket_lock:
            self.client_socket.sendall(RSMetaMsg(RSMetaType.CLIENT_PUT, len(bs)).serialize())
            self.client_socket.sendall(bs)

    def get(self, query: str) -> list:
        bs = query.encode('utf-8')
        with self.socket_lock:
            # send to server
            self.client_socket.sendall(RSMetaMsg(RSMetaType.CLIENT_GET, len(bs)).serialize())
            self.client_socket.sendall(bs)
            # receive from server
            return_meta = self.client_socket.recv(RSMetaMsg.get_size())
            return_meta = RSMetaMsg.deserialize(return_meta)
            data = self.client_socket.recv(return_meta.meta_len).decode('utf-8')
            # split the data by the delimiter
            data = data.split(RSMsgDelimiter.RSJOINBLOCK)
            # remove the last empty string if it exists
            if data[-1] == '':
                data = data[:-1]
        return data
    
    def save_db(self):
        with self.socket_lock:
            # send to server
            self.client_socket.sendall(RSMetaMsg(RSMetaType.CLIENT_SAVE).serialize())
            # TODO: check the return value
        