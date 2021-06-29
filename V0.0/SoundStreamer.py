import pyaudio
import struct
import math
import wave
import time
import numpy
import scipy.io.wavfile as wav
from npsocket import SocketNumpyArray
import datetime

from SoundClient import AudioClient

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100 
INPUT_BLOCK_TIME = 0.05
#INPUT_FRAMES_PER_BLOCK = int(RATE*INPUT_BLOCK_TIME)
INPUT_FRAMES_PER_BLOCK = 1024

class Soundstreamer(object):
    def __init__(self):
        self.fname = "test2.wav"
        self.mode ='wb'
        self.pa = pyaudio.PyAudio()
        self.wavefile = self._prepare_file(self.fname, self.mode)
        self.list_all_devices()
        self.eingabe_auswahl()
        self.frames = []
        #self.AudioClient = AudioClient(50000,"localhost")
        #self.AudioClient.connect_to_server()
        self.sock_sender = SocketNumpyArray()
        self.sock_sender.initialize_sender('localhost', 9999)
        f = open("sendetimes.txt", "w")
        f.write("")
        f.close()
        #self.stream = self.open_mic_stream()
        self.stream = self.open_mic_stream_without_callback()

    def eingabe_auswahl(self):
        print("Geben sie ihre Device Id für das Mikrofon ein: ")
        device_id = input()
        self.device_id = int(device_id)

    def list_all_devices(self):
        Ausgabedevices = []
        Eingabedevices = []

        for i in range( self.pa.get_device_count() ):    
            Deviceeigenschaften = [] 
            devinfo = self.pa.get_device_info_by_index(i)
            for keyword in ["microfon","input","mikrofon"]:   
                if keyword in devinfo["name"].lower():
                    print( "Device %d: %s"%(i,devinfo["name"]) )
                    Deviceeigenschaften = [i,devinfo]
                    Eingabedevices.append(Deviceeigenschaften)
            for keyword in ["output","headphones","kopfhörer","lautsprecher","speaker","nvidia"]:
                if keyword in devinfo["name"].lower():
                    print( "Device %d: %s"%(i,devinfo["name"]) )
                    Deviceeigenschaften = [i,devinfo]
                    Ausgabedevices.append(Deviceeigenschaften)

        print("Folgende Ausgabe Devices gefunden:")
        for device in Ausgabedevices:
            print("Device ",device[0]," ",device[1]["name"])

        print("-"*50)
        print("Folgende Eingabe Devices gefunden:")
        for device in Eingabedevices:
            print("Device ",device[0]," ",device[1]["name"])

        self.Eingabedevices = Eingabedevices
        self.Ausgabedevices = Ausgabedevices


    def open_mic_stream(self):
        stream = self.pa.open(   format = FORMAT,
                                 channels = CHANNELS,
                                 rate = RATE,
                                 input = True,
                                 input_device_index = self.device_id,
                                 frames_per_buffer = INPUT_FRAMES_PER_BLOCK,
                                 stream_callback=self.get_callback())
        return(stream)

    def open_mic_stream_without_callback(self):
        stream = self.pa.open(   format = FORMAT,
                                 channels = CHANNELS,
                                 rate = RATE,
                                 input = True,
                                 input_device_index = self.device_id,
                                 frames_per_buffer = INPUT_FRAMES_PER_BLOCK)
        return(stream)
    def start_recording(self):
        self.stream.start_stream()

    def stop_recording(self):
        self.stream.stop_stream()
        #return self

    def get_callback(self):
        def callback(in_data, frame_count, time_info, status):
            self.wavefile.writeframes(in_data)
            return in_data, pyaudio.paContinue
        return callback

      
    def stop_mic_stream(self):
        self.stream.close()

    def listen_stream(self):
        try:
            block = self.stream.read(INPUT_FRAMES_PER_BLOCK)
        except IOError as e:
            # dammit. 
            print( "(%d) Error recording: %s"%(1,e) )
            return
        #print(block)
        test = numpy.fromstring(block,dtype=numpy.int16)
        self.frames.append(numpy.fromstring(block, dtype=numpy.int16))
        self.send_to_Server(numpy.fromstring(block, dtype=numpy.int16))

    def _prepare_file(self, fname, mode='wb'):
        wavefile = wave.open(fname, mode)
        wavefile.setnchannels(CHANNELS)
        wavefile.setsampwidth(self.pa.get_sample_size(pyaudio.paInt16))
        wavefile.setframerate(RATE)
        return wavefile

    def stream_to_file(self):
        numpydata = numpy.hstack(self.frames)
        wav.write('out.wav',RATE,numpydata)
        self.send_to_Server(numpydata)

    def send_to_Server(self,daten):
        #numpydata = numpy.hstack(self.frames)
        #daten = numpy.fromstring(daten, dtype=numpy.int16)
        #daten.tostring()
        #self.AudioClient.send_data_to_server(daten)
        
        sendetime = datetime.datetime.now()
        f = open("sendetimes.txt", "a")
        f.write(str(sendetime)+"\n")
        f.close()

        self.sock_sender.send_numpy_array(daten)
    
    def stop_audio(self):
        self.AudioClient.close_connection()

    def send_states(self):
        self.sock_sender.send_numpy_array(["stopping"])

if __name__ == "__main__":
    BOB = Soundstreamer()
    #BOB.list_all_devices()
    for i in range(1000):
        #BOB.send_states()
        BOB.listen_stream()
    BOB.send_states()
    #BOB.stream_to_file()
    #BOB.stop_audio()

    #BOB.start_recording()
    #time.sleep(20)
    #BOB.stop_recording()
