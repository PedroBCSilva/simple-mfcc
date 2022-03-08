import matplotlib.pyplot as plt
from python_speech_features import mfcc
import sounddevice as sd
import scipy.io.wavfile as wav
import librosa
import torchaudio.transforms
import torch
import numpy as np

DEFAULT_INPUT_DEVICE = 'Stereo Mix (Realtek(R) Audio), MME'
DEFAULT_MFCC_IMAGE_NAME = 'mfcc-{}'
SAMPLE_RATE = 48000
RECORDING_WINDOW_INTERVAL = 1


n_mfcc = 13
n_mels = 20
n_fft = 512
hop_length = 160
fmin = 0
fmax = None

melkwargs={"n_fft" : n_fft, "n_mels" : n_mels, "hop_length":hop_length, "f_min" : fmin, "f_max" : fmax}


# print(sd.query_devices())

def load_sample_audio():
    (sample_rate, sig) = wav.read("dog-bark.wav")
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


def create_torch_mfcc(record, sample_rate):
    return torchaudio.transforms.MFCC(sample_rate=sample_rate,
                                      n_mfcc=n_mfcc,
                                      dct_type=2,
                                      norm='ortho',
                                      log_mels=False,
                                      melkwargs=melkwargs).cuda()(torch.from_numpy(record))


def create_librosa_mfcc(record, sample_rate):
    (y, sr) = librosa.load("dog-bark.wav")
    S = librosa.feature.melspectrogram(y=y, sr=sr).T
    return librosa.feature.mfcc(y,sr, n_mfcc=40,n_fft=1024, S=S.T,
                                             win_length=int(0.025*sr),
                                             hop_length=int(0.01*sr),htk=False)
    # return librosa.feature.mfcc(y, sr, n_mfcc=96,
    #                                          n_fft=1024,
    #                                          win_length=int(0.025*sr),
    #                                          hop_length=int(0.01*sr))


def create_mfcc(record, sample_rate):
    mfcc_feat = mfcc(record, sample_rate, appendEnergy=True,
                     winlen=0.025,
                     winstep=0.01,
                     numcep=40,
                     nfilt=80,
                     nfft=1024)
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
    (sig_librosa, sample_rate_librosa) = librosa.load('StarWars3.wav')
    # python speech features
    python_speech_mfcc = create_mfcc(sig, sample_rate)
    plot_and_save_mfcc(python_speech_mfcc, 'mfcc-speech-dog')
    #librosa
    librosa_features = create_librosa_mfcc(sig_librosa, sample_rate_librosa)
    plot_and_save_mfcc(librosa_features, 'mfcc-librosa-dog')
    # torch audio
    # torch_mfcc = create_torch_mfcc(sig, sample_rate)
    # plot_and_save_mfcc(torch_mfcc, 'mfcc-torch')


if __name__ == "__main__":
    main()
