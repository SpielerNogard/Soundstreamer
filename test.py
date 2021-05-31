import sounddevice as sd
duration = 5  # seconds

def callback(indata, outdata, frames, time, status):
    if status:
        print(status)
    outdata[:] = indata

with sd.RawStream(channels=2, dtype='int24', callback=callback):
    print("1")
    sd.sleep(int(duration * 1000))

test = sd.Stream.read(channels=2, callback=callback)
print(str(test))
