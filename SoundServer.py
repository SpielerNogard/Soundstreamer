#from SoundClient import AudioClient
import socket  # Import socket module



class AudioServer():
    def __init__(self,port,hostname):
        self.port = port
        self.host = hostname
        self.Server = socket.socket()

    def start_server(self):
        self.Server.bind(('localhost',self.port))#Bind to laclhost
        self.Server.listen(5)
        print("Server started")

        while True:
            self.conn,self.address = self.Server.accept() #connect to client

            while True:
                print("Connected to: ",self.address)
                self.data = self.conn.recv(1024)
                print("Data recieved", self.data)
                self.send_to_connection()

    def send_to_connection(self):
        st = 'Thank you for connecting'
        byt = st.encode()
        self.conn.send(byt)

BOB = AudioServer(50000,"Server")
BOB.start_server()


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
                print('Server received', data)
    
                st = 'Thank you for connecting'
                byt = st.encode()
                conn.send(byt)
    
                x += 1
    
            except Exception as e:
                print(e)
                break

#run()
    