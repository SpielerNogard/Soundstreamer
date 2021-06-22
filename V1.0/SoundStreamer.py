import pyaudio
#import wave
import numpy
#import scipy.io.wavfile as wav
from npsocket import SocketNumpyArray
import datetime

class SoundStreamer(object):
    def __init__(self,port,adress):
        self.port = port 
        self.adress = adress
        self.fname = "SoundFiles/Client.wav"
        self.mode ='wb'
        self.frames = []

        self.pa = pyaudio.PyAudio()

        #Stream data for pyaudio
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100 
        self.INPUT_FRAMES_PER_BLOCK = 1024
        self.create_server_sender()
        self.list_all_devices()
        self.eingabe_auswahl()
        self.stream = self.open_stream_without_callback()

    def create_server_sender(self):
        self.sock_sender = SocketNumpyArray()
        self.sock_sender.initialize_sender(self.adress,self.port)

    def set_file_name(self,filename):
        self.fname = "SoundFiles/"+filename+".wav"

    def open_stream_without_callback(self):
        stream = self.pa.open(   format = self.FORMAT,
                                 channels = self.CHANNELS,
                                 rate = self.RATE,
                                 input = True,
                                 input_device_index = self.device_id,
                                 frames_per_buffer = self.INPUT_FRAMES_PER_BLOCK)
        return(stream)

    def stop_stream(self):
        self.stream.close()

    def listen_stream(self):
        try:
            block = self.stream.read(self.INPUT_FRAMES_PER_BLOCK)
        except IOError as e:
            # dammit. 
            print( "(%d) Error recording: %s"%(1,e) )
            return

        #self.frames.append(numpy.fromstring(block, dtype=numpy.int16))
        self.send_to_Server(numpy.fromstring(block, dtype=numpy.int16))
        #self.write_soundfile()

    def write_soundfile(self):
        numpydata = numpy.hstack(self.frames)
        #wav.write(self.fname,self.RATE,numpydata)
        
    def stop_recording(self):
        self.send_to_Server(["stopping"])
        

    def send_to_Server(self,daten):
        self.sock_sender.send_numpy_array(daten)

    def create_time_stamp(self):
        sendetime = datetime.datetime.now()
        f = open("TimeStamps/ClientTimeStamps.txt", "a")
        f.write(str(sendetime)+"\n")
        f.close()

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

    def eingabe_auswahl(self):
        print("Geben sie ihre Device Id für das Mikrofon ein: ")
        device_id = input()
        self.device_id = int(device_id)


if __name__ == "__main__":
    CLAUS = SoundStreamer(9999,"192.168.10.22")
    for i in range(1000):
        CLAUS.listen_stream()
        CLAUS.create_time_stamp()
        print(i)

    CLAUS.stop_recording()