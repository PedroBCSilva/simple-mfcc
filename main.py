import matplotlib.pyplot as plt
from python_speech_features import mfcc
import scipy.io.wavfile as wav

(sample_rate, sig) = wav.read("StarWars3.wav")

mfcc_feat = mfcc(sig, sample_rate)
mfcc_data = mfcc_feat.T

plt.figure(figsize=(25, 8))
plt.imshow(mfcc_data, cmap=plt.cm.jet, aspect='auto', origin='lower')
plt.title('Star Wars MFCC', fontsize=18)
plt.xlabel('Time [s]', fontsize=18)
plt.ylabel('MFCC', fontsize=18)
plt.show()
