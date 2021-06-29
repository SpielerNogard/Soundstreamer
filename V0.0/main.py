import sounddevice as sd
Ergebnis = sd.query_devices()
for a in Ergebnis: 
    print(a)




from scipy.io.wavfile import write

fs = 44100  # Sample rate
seconds = 3  # Duration of recording
sd.default.device = 1
myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
sd.wait()  # Wait until recording is finished
write('output.wav', fs, myrecording)  # Save as WAV file 