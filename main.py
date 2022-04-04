import threading

import matplotlib.pyplot as plt
import sounddevice as sd
import scipy.io.wavfile as wav
import librosa
import librosa.display as librosa_display
import configurations as conf
from threading import Thread

# print(sd.query_devices())


def load_sample_audio():
    (sample_rate, sig) = load_sample_audio()
    return sample_rate, sig


def setup_sound_device():
    sd.default.device = conf.DEFAULT_INPUT_DEVICE


def record_window():
    recorded_window = sd.rec(int(conf.RECORDING_WINDOW_INTERVAL * conf.SAMPLE_RATE), samplerate=conf.SAMPLE_RATE, channels=1)
    sd.wait()  # Wait until recording is finished
    return recorded_window.flatten()


def create_mfcc(record, sample_rate):
    features = librosa.feature.mfcc(record, sample_rate, n_fft=conf.N_FFT,
                                    n_mfcc=conf.N_MFCC, n_mels=conf.N_MELS,
                                    hop_length=conf.HOP_LENGTH,
                                    fmin=conf.MIN_FREQ, fmax=conf.MAX_FREQ, htk=False)
    return features


def plot_and_save_mfcc(mfcc_data, file_name, sample_rate):
    plt.figure(figsize=(10, 8))
    librosa_display.specshow(mfcc_data, sr=sample_rate, x_axis='time', cmap='coolwarm')
    plt.savefig(file_name)
    plt.cla()


def save_record(file_name, recording, sample_rate):
    wav.write(file_name + '.wav', sample_rate, recording)


def debug_save_mfcc(mfcc_data, record, sample_rate, file_name):
    print('start save for ', file_name)
    plot_and_save_mfcc(mfcc_data, file_name, sample_rate)
    save_record(file_name, record, sample_rate)
    print('finish save for ', file_name)
    return 0

# TODO: Solve high memory issue since thread exiting is not releasing memory
def start_listening_and_creating_mfcc():
    image_count = 0
    while True:
        current_window = record_window()
        mfcc_data = create_mfcc(current_window, conf.SAMPLE_RATE)
        save_thread = Thread(target=debug_save_mfcc,
                             args=(mfcc_data,
                                   current_window,
                                   conf.SAMPLE_RATE,
                                   conf.DEFAULT_MFCC_IMAGE_NAME.format(image_count)
                                   ),
                             daemon=True
                             )
        save_thread.start()
        print(f"THREADS: {len(threading.enumerate())}")
        image_count += 1


# This method can be used for testing purpose of single file mfcc
def create_mfcc_from_file(file_path):
    (signal, sample_rate) = librosa.load(file_path)
    librosa_features = create_mfcc(signal, sample_rate)
    plot_and_save_mfcc(librosa_features, 'mfcc-librosa', sample_rate)


def main():
    print('##### Starting recording system audio #####')
    setup_sound_device()
    start_listening_and_creating_mfcc()
    # create_mfcc_from_file(conf.SAMPLE_FILE_NAME)


if __name__ == "__main__":
    main()
