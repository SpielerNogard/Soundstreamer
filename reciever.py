from npsocket import SocketNumpyArray
import scipy.io.wavfile as wav
import pickle
import pyaudio
import numpy as np
import datetime
RATE = 44100 
BITRATE = 44100 #number of frames per second/frameset. 

p = pyaudio.PyAudio()
stream = p.open(format = pyaudio.paInt16, 
            channels = 1, 
            rate = BITRATE, 
            output = True)

sock_receiver = SocketNumpyArray()
sock_receiver.initalize_receiver(9999) # the 9999 is the port you can change it with your own. 
f = open("recievetimes.txt", "w")
f.write("")
f.close()
while True:
    frame = sock_receiver.receive_array()  # Receiving the image as numpy array. 

    sendetime = datetime.datetime.now()
    f = open("recievetimes.txt", "a")
    f.write(str(sendetime)+"\n")
    f.close()

    #print(frame)
    wav.write('out_server.wav',RATE,frame)
    stream.write(frame.astype(np.int16).tostring())

stream.stop_stream()
stream.close()
p.terminate()