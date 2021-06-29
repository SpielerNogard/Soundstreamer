import pyaudio
#import wave
import wavio
import numpy
#import soundfile as sf
from npsocket import SocketNumpyArray
import datetime

class SoundStreamer(object):
    def __init__(self,port,adress,sample):
        self.port = port 
        self.adress = adress
        self.fname = "SoundFiles/_48000_"+str(sample)+"_w.wav"
        self.mode ='wb'
        self.frames = []

        self.pa = pyaudio.PyAudio()

        #Stream data for pyaudio
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 48000 
        self.INPUT_FRAMES_PER_BLOCK = sample
        #self.create_server_sender()
        self.list_all_devices()
        self.eingabe_auswahl()
        #self.device_id = 2
        self.stream = self.open_stream_without_callback()

        self.stream2 = self.pa.open(format = pyaudio.paInt16, 
                channels = 1, 
                rate = self.RATE, 
                output = True)

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
        self.create_time_stamp()
        try:
            block = self.stream.read(self.INPUT_FRAMES_PER_BLOCK)
        except IOError as e:
            # dammit. 
            print( "(%d) Error recording: %s"%(1,e) )
            return

        self.frames.append(numpy.fromstring(block, dtype=numpy.int16))
        #self.send_to_Server(numpy.fromstring(block, dtype=numpy.int16))
        #self.write_soundfile()
        self.play_file(numpy.fromstring(block, dtype=numpy.int16))
        self.create_time_stamp2()

    def write_soundfile(self):
        numpydata = numpy.hstack(self.frames)
        #sf.write(self.fname, numpydata, self.RATE)
        wavio.write(self.fname, numpydata, self.RATE, sampwidth=2)
        
    def stop_recording(self):
        self.send_to_Server(["stopping"])
        

    def send_to_Server(self,daten):
        self.sock_sender.send_numpy_array(daten)

    def create_time_stamp(self):
        sendetime = datetime.datetime.now()
        f = open("TimeStamps/ClientTimeStamps_before_output.txt", "a")
        f.write(str(sendetime)+"\n")
        f.close()

    def create_time_stamp2(self):
        sendetime = datetime.datetime.now()
        f = open("TimeStamps/ClientTimeStamps_after_output.txt", "a")
        f.write(str(sendetime)+"\n")
        f.close()
    def list_all_devices(self):
        Ausgabedevices = []
        Eingabedevices = []

        for i in range( self.pa.get_device_count() ):    
            Deviceeigenschaften = [] 
            devinfo = self.pa.get_device_info_by_index(i)
            print( "Device %d: %s"%(i,devinfo["name"]) )
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

    def play_file(self,frame):
        self.stream2.write(frame.astype(numpy.int16).tostring())

if __name__ == "__main__":
    samples = [512]
    for sample in samples:
        CLAUS = SoundStreamer(9999,"192.168.10.22",sample)
        for i in range(int((4096/sample)*200)):
            CLAUS.listen_stream()
            #CLAUS.create_time_stamp()
            print(i)
        CLAUS.write_soundfile()

    #CLAUS.stop_recording()