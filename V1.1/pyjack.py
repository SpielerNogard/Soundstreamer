import pyaudio
import wavio
import numpy
import pyfiglet

import config

class PyJack_Client(object):
    def __init__(self):
        self.write_logo()
        #my Filename to save Soundfile
        self.fname = ""
        self.get_filename()
        #place to store samples
        self.frames = []
        self.pa = pyaudio.PyAudio()
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = config.CHANNELS
        self.RATE = config.SAMPLE_RATE
        self.INPUT_FRAMES_PER_BLOCK = config.CHUNK_SIZE
        self.is_recording = False

        self.create_capture_stream()
        self.create_output_stream()
        self.start_recording()
        #self.recording()

    def write_logo(self):
        ascii_banner = pyfiglet.figlet_format("PYJ(H)ack")
        print(ascii_banner)
    def get_filename(self):
        #UserInput for his Filename
        wish_filename = input("Bitte geben sie den Namen für ihr Soundfile ein (ohne Dateiendung): ")
        self.fname = "SoundFiles/"+wish_filename+".wav"
        print("Ok. Ihre Datei wird unter ",self.fname," gespeichert")

    def list_all_devices(self):
        for i in range( self.pa.get_device_count() ):    
            Deviceeigenschaften = [] 
            devinfo = self.pa.get_device_info_by_index(i)
            print( "Device %d: %s"%(i,devinfo["name"]))

    def create_capture_stream(self):
        self.list_all_devices()
        device_id = int(input("Bitte geben sie ihre Device Id an (Input): "))
        self.capture_stream = self.pa.open(   format = self.FORMAT,
                                 channels = self.CHANNELS,
                                 rate = self.RATE,
                                 input = True,
                                 input_device_index = device_id,
                                 frames_per_buffer = self.INPUT_FRAMES_PER_BLOCK)
        
    def create_output_stream(self):
        self.output_stream = self.pa.open(format = pyaudio.paInt16, 
                channels = 1, 
                rate = self.RATE, 
                output = True)

    def start_recording(self):
        print("Recording started: Press CTRL+C to save your File")
        self.is_recording = True
        while self.is_recording==True:
            self.recording()

    def recording(self):
        
        try:
            block = self.capture_stream.read(self.INPUT_FRAMES_PER_BLOCK)
        except IOError as e:
            # dammit. 
            print( "(%d) Error recording: %s"%(1,e) )
            return
        try:
            #saving the sample into an array
            self.frames.append(numpy.frombuffer(block, dtype=numpy.int16))
            if config.PLAYING == True:
                self.play_sample(numpy.frombuffer(block, dtype=numpy.int16))
        except KeyboardInterrupt:
            self.is_recording = False
            self.save_file()
    
    def save_file(self):
        print("Saving your Sounfile ....")
        numpydata = numpy.hstack(self.frames)
        wavio.write(self.fname, numpydata, self.RATE, sampwidth=2)
        print("done")
        
    def play_sample(self,block):
        self.output_stream.write(block.astype(numpy.int16).tobytes())

if __name__ == "__main__":
    BOB = PyJack_Client()
