import matplotlib.pyplot as plt
from python_speech_features import mfcc
import sounddevice as sd
from scipy.io.wavfile import write

fs = 44100  # Sample rate
seconds = 5  # Duration of recording
sd.default.device = 'Loopback Audio 2, Core Audio'
plt.figure(figsize=(25, 8))
plt.title('Current audio MFCC', fontsize=18)
plt.xlabel('Time [s]', fontsize=18)
plt.ylabel('MFCC', fontsize=18)
imageCount = 0
imageName = 'mfcc-{}'

while True:
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished
    mfcc_feat = mfcc(myrecording, fs)
    mfcc_data = mfcc_feat.T
    plt.imshow(mfcc_data, cmap=plt.cm.jet, aspect='auto', origin='lower')
    # plt.show()
    plt.savefig(imageName.format(imageCount))
    write(imageName.format(imageCount)+'.wav',fs,myrecording)
    imageCount+=1
