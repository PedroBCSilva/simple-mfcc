import matplotlib.pyplot as plt
from python_speech_features import mfcc
import sounddevice as sd
from scipy.io.wavfile import write

DEFAULT_INPUT_DEVICE = 'Stereo Mix (Realtek(R) Audio), MME'
DEFAULT_MFCC_IMAGE_NAME = 'mfcc-{}'
SAMPLE_RATE = 48000
RECORDING_WINDOW_INTERVAL = 5


# print(sd.query_devices())


def setup_sound_device():
    sd.default.device = DEFAULT_INPUT_DEVICE
    plt.figure(figsize=(25, 8))
    plt.title('Current audio MFCC', fontsize=18)
    plt.xlabel('Time [s]', fontsize=18)
    plt.ylabel('MFCC', fontsize=18)


def record_window():
    recorded_window = sd.rec(int(RECORDING_WINDOW_INTERVAL * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=2)
    sd.wait()  # Wait until recording is finished
    return recorded_window


def start_listening_and_creating_mfcc():
    image_count = 0
    while True:
        my_recording = record_window()
        mfcc_feat = mfcc(my_recording, SAMPLE_RATE)
        mfcc_data = mfcc_feat.T
        plt.imshow(mfcc_data, cmap=plt.cm.jet, aspect='auto', origin='lower')
        # plt.show()
        plt.savefig(DEFAULT_MFCC_IMAGE_NAME.format(image_count))
        write(DEFAULT_MFCC_IMAGE_NAME.format(image_count) + '.wav', SAMPLE_RATE, my_recording)
        image_count += 1


def main():
    setup_sound_device()
    start_listening_and_creating_mfcc()


if __name__ == "__main__":
    main()
