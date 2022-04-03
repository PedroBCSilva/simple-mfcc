import matplotlib.pyplot as plt
import sounddevice as sd
import scipy.io.wavfile as wav
import librosa
import librosa.display as librosa_display
import configurations as conf

# print(sd.query_devices())


def load_sample_audio():
    (sample_rate, sig) = load_sample_audio()
    return sample_rate, sig


def setup_sound_device():
    sd.default.device = conf.DEFAULT_INPUT_DEVICE


def record_window():
    recorded_window = sd.rec(int(conf.RECORDING_WINDOW_INTERVAL * conf.SAMPLE_RATE), samplerate=conf.SAMPLE_RATE, channels=2)
    sd.wait()  # Wait until recording is finished
    return recorded_window


def create_mfcc(record, sample_rate):
    features = librosa.feature.mfcc(record, sample_rate, n_fft=conf.N_FFT,
                                    n_mfcc=conf.N_MFCC, n_mels=conf.N_MELS,
                                    hop_length=conf.HOP_LENGTH,
                                    fmin=conf.MIN_FREQ, fmax=conf.MAX_FREQ, htk=False)
    return features


def plot_and_save_mfcc(mfcc_data, file_name, sample_rate):
    plt.figure(figsize=(10, 8))
    plt.title('Current audio MFCC', fontsize=18)
    plt.xlabel('Time [s]', fontsize=18)
    librosa_display.specshow(mfcc_data, sr=sample_rate, x_axis='time', cmap='coolwarm')
    plt.savefig(file_name)
    plt.cla()


def start_listening_and_creating_mfcc():
    image_count = 0
    while True:
        my_recording = record_window()
        mfcc_data = create_mfcc(my_recording, conf.SAMPLE_RATE)
        plot_and_save_mfcc(mfcc_data, conf.DEFAULT_MFCC_IMAGE_NAME.format(image_count), conf.SAMPLE_RATE)
        wav.write(conf.DEFAULT_MFCC_IMAGE_NAME.format(image_count) + '.wav', conf.SAMPLE_RATE, my_recording)
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
