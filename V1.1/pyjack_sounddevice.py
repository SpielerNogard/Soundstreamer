import sounddevice as sd
import wavio

fs = 48000 # Sample rate
seconds = 3  # Duration of recording

myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()  # Wait until recording is finished

wavio.write('SoundFiles/soundevice.wav', myrecording,fs, sampwidth=1)