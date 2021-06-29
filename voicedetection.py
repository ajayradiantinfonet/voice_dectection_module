import sys
import time
import sounddevice as sd
import numpy as np 
import webrtcvad

channels = [1]
mapping  = [c - 1 for c in channels]

device_info = sd.query_devices(None, 'input')
sample_rate = 16000

interval_size = 30 
downsample = 1

block_size = sample_rate * interval_size / 1000

vd = webrtcvad.Vad()

print("Reading audio stream :\n" + str(sd.query_devices()) + '\n')
print(F"audio input channels to process: {channels}")
print(F"sample_rate: {sample_rate}")
print(F"window size: {interval_size} ms")
print(F"datums per window: {block_size}")
print()


def voice_detection(audio_data):
    return vd.is_speech(audio_data, sample_rate)


def audio_callback(indata, frames, time, status):
    if status:
        print(F"underlying audio stack warning:{status}", file=sys.stderr)

    assert frames == block_size
    audio_data = indata[::downsample, mapping]        
    audio_data = map(lambda x: (x+1)/2, audio_data)   
    audio_data = np.fromiter(audio_data, np.float16)  


    audio_data = audio_data.tobytes()
    detection = voice_detection(audio_data)
    print(f'{detection} \r', end="") 


with sd.InputStream(
    device=None,  
    channels=max(channels),
    samplerate=sample_rate,
    blocksize=int(block_size),
    callback=audio_callback):

    while True:
        time.sleep(0.1)  