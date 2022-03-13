import matplotlib.pyplot as plt
from python_speech_features import mfcc
import sounddevice as sd
import scipy.io.wavfile as wav
from scipy.signal.windows import hann
import librosa
import librosa.display as librosa_display
import numpy as np

DEFAULT_INPUT_DEVICE = 'Stereo Mix (Realtek(R) Audio), MME'
DEFAULT_MFCC_IMAGE_NAME = 'mfcc-{}'
SAMPLE_RATE = 48000
RECORDING_WINDOW_INTERVAL = 1
SAMPLE_FILE_NAME= "dog-bark.wav"

n_mfcc = 40
n_mels = 40
n_fft = 512
hop_length = 160
fmin = 0
fmax = None

# print(sd.query_devices())

def load_sample_audio():
    (sample_rate, sig) = wav.read(SAMPLE_FILE_NAME)
    return sample_rate, sig


def setup_sound_device():
    sd.default.device = DEFAULT_INPUT_DEVICE
    # plt.figure(figsize=(25, 8))
    # plt.title('Current audio MFCC', fontsize=18)
    # plt.xlabel('Time [s]', fontsize=18)
    # plt.ylabel('MFCC', fontsize=18)


def record_window():
    recorded_window = sd.rec(int(RECORDING_WINDOW_INTERVAL * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=2)
    sd.wait()  # Wait until recording is finished
    return recorded_window


def create_librosa_mfcc(record, sample_rate):
    features = librosa.feature.mfcc(record, sample_rate,  n_fft=n_fft,
                                    n_mfcc=n_mfcc, n_mels=n_mels,
                                    hop_length=hop_length,
                                    fmin=fmin, fmax=fmax, htk=False)
    return features


def create_mfcc(record, sample_rate):
    mfcc_feat = mfcc(record, sample_rate,winlen=n_fft / sample_rate, winstep=hop_length / sample_rate,
                                          numcep=n_mfcc, nfilt=n_mels, nfft=n_fft, lowfreq=fmin, highfreq=fmax,
                                          preemph=0.0, ceplifter=0, appendEnergy=False, winfunc=hann)
    return mfcc_feat.T


def plot_and_save_mfcc(mfcc_data, file_name):
    plt.figure(figsize=(10, 8))
    plt.title('Current audio MFCC', fontsize=18)
    plt.xlabel('Time [s]', fontsize=18)
    plt.ylabel('MFCC', fontsize=18)
    plt.imshow(mfcc_data, aspect='auto', origin='lower')
    # plt.show()
    plt.savefig(file_name)


def start_listening_and_creating_mfcc():
    image_count = 0
    while True:
        my_recording = record_window()
        mfcc_data = create_mfcc(my_recording, SAMPLE_RATE)
        plot_and_save_mfcc(mfcc_data, DEFAULT_MFCC_IMAGE_NAME.format(image_count))
        wav.write(DEFAULT_MFCC_IMAGE_NAME.format(image_count) + '.wav', SAMPLE_RATE, my_recording)
        image_count += 1


def main():
    print('##### Starting recording system audio #####')
    # setup_sound_device()
    # start_listening_and_creating_mfcc()
    (sample_rate, sig) = load_sample_audio()
    (sig_librosa, sample_rate_librosa) = librosa.load(SAMPLE_FILE_NAME)
    # python speech features
    python_speech_mfcc = create_mfcc(sig, sample_rate)
    plot_and_save_mfcc(python_speech_mfcc, 'mfcc-speech')
    #librosa
    librosa_features = create_librosa_mfcc(sig_librosa, sample_rate_librosa)
    # plot_and_save_mfcc(librosa_features, 'mfcc-librosa')
    librosa_display.specshow(librosa_features, sr=sample_rate_librosa, x_axis='time')
    plt.savefig('MFCCs.png')

if __name__ == "__main__":
    main()
