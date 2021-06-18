#from SoundClient import AudioClient
import socket  # Import socket module
import numpy as np
import scipy.io.wavfile as wav
import pickle
RATE = 44100 
class AudioServer():
    def __init__(self,port,hostname):
        self.port = port
        self.host = hostname
        self.ready_to_connect = True
        self.Server = socket.socket()

    def start_server(self):
        self.Server.bind(('localhost',self.port))#Bind to laclhost
        self.Server.listen(5)
        print("Server started")
        data = b""
        while True:
            self.conn,self.address = self.Server.accept() #connect to client

            while self.ready_to_connect == True:
                print("Connected to: ",self.address)
                packet = self.conn.recv(1024)
                print("Data recieved", packet)
                if not packet: break
                data += packet
                #self.send_to_connection()
                
            self.recieve_data(data)
                    
                
                
                    
            self.write_to_audiofile()
                    #self.ready_to_connect = False
            self.Server.shutdown()
            self.Server.close()

    def write_to_audiofile(self):
        wav.write('out_server.wav',RATE,self.Daten)

    def recieve_data(self,data):
        frame = pickle.loads(data)
        try:
            self.Daten += frame
        except AttributeError:
            self.Daten = frame
        
        

    def send_to_connection(self):
        st = 'Thank you for connecting'
        byt = st.encode()
        self.conn.send(byt)

BOB = AudioServer(50000,"Server")
BOB.start_server()

TEST = []
def run():   
    port = 50000  # Reserve a port for your service every new transfer wants a new port or you must wait.
    s = socket.socket()  # Create a socket object
    host = ""  # Get local machine name
    s.bind(('localhost', port))  # Bind to the port
    s.listen(5)  # Now wait for client connection.
    
    print('Server listening....')
    
    x = 0
    
    while True:
        conn, address = s.accept()  # Establish connection with client.
    
        while True:
            try:
                print('Got connection from', address)
                data = conn.recv(1024)
                Baum = np.fromstring(data)
                TEST.append(Baum)
                #print('Server received', data)
                print(x)
                #st = 'Thank you for connecting'
                #byt = st.encode()
                #conn.send(byt)
    
                x += 1

                if x == 1000:
                    numpydata = np.hstack(TEST)
                    wav.write('out_server.wav',RATE,numpydata)
    
            except Exception as e:
                print(e)
                break

#run()
    