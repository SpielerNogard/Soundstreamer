from npsocket import SocketNumpyArray
import scipy.io.wavfile as wav
import pickle
import pyaudio
import numpy as np
import datetime
import soundfile as sf



class SoundServer(object):
    def __init__(self,port,filename):
        self.port = port
        self.state = "receiving"
        self.RATE = 44100
        self.filename = "SoundFiles/"+filename+".wav"
        self.frames = []
        self.create_server()
        self.run()

    def create_server(self):
        self.sock_receiver = SocketNumpyArray()
        self.sock_receiver.initalize_receiver(self.port)

    def run(self):
        while True:
            frame = self.sock_receiver.receive_array() 
            if frame == ["stopping"]:
                    print("stopping server")
                    self.state = "stopping"

            if self.state == "receiving":
                #self.frames.append(frame)
                #self.write_to_wav_file()
                print(frame)
            elif self.state == "stopping":
                break
            self.create_time_stamp()
    def write_to_wav_file(self):
        numpydata = np.hstack(self.frames)
        #wav.write(self.filename,self.RATE,numpydata)
        #sf.write(self.filename, numpydata, self.RATE)

    def create_time_stamp(self):
        sendetime = datetime.datetime.now()
        f = open("TimeStamps/ServerTimeStamps.txt", "a")
        f.write(str(sendetime)+"\n")
        f.close()

if __name__ == "__main__":
    CLAUS = SoundServer(9999,"test_new_server")