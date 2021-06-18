import socket  # Import socket module
import os
import re
import pickle

class DataPacket:
    def __init__(self):
        pass

    def write(self, string):
        self._string = string

    @property
    def string(self):
        return self._string

class AudioClient():
    def __init__(self,port,Serverip):
        self.port = port
        self.Serverip = Serverip
        self.Socket = socket.socket()

    def connect_to_server(self):
        self.Socket.connect((self.Serverip,self.port))

    def send_data_to_server(self,data):
        
        data = pickle.dumps(data)
        self.Socket.send(data)

    def recieve_data(self):
        data = self.Socket.recv(1024)
        if data:
            print(data)
        else:
            print("no data recieved")

    def close_connection(self):
        self.Socket.shutdown()
        self.Socket.close()

#BOB = AudioClient(50000,"localhost")    
#BOB.connect_to_server()
#x = 0
#while x <100:
    #data = "hi"
    #byt = data.encode()
    #BOB.send_data_to_server(byt)
    #BOB.recieve_data()
    #x+=1
        


