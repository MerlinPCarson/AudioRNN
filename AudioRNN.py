import numpy as np
import soundfile as sf
import librosa
import argparse
import pickle
import glob, os, time, math
from scipy import signal
from audioop import lin2ulaw, ulaw2lin

SAMPLERATE = 16000
EMPHASISCOEFF = -0.95

TO_ULAW_SCALE = 255.0/32767.0
FROM_ULAW_SCALE = TO_ULAW_SCALE ** -1

def from_ulaw(samples):
    dec_samples = []
    for sample in samples:
        ampl_val_8 = ((((sample) / 255.0) - 0.5) * 2.0)
        ampl_val_16 = (np.sign(ampl_val_8) * (1/256.0) * ((1 + 256.0)**abs(ampl_val_8) - 1)) * 2**15
        dec_samples.append(ampl_val_16)
    return np.array(dec_samples, dtype=np.int16)

def to_ulaw(samples):
    enc_samples = [int((np.sign(sample) * (np.log(1 + 256*abs(sample)) / (
            np.log(1+256))) + 1)/2.0 * 255) for sample in samples]
    return np.array(enc_samples, dtype=np.uint8)

def write_audio(samples, file_name):

    sf.write(file_name, samples, SAMPLERATE, subtype='PCM_16')
    print('Audio saved to disk.')

def load_data(datadir):

    concat_data = np.array([],dtype='float32')

    for waveFile in glob.glob(os.path.join(datadir, '**', '*.wav'),recursive=True):
        print('loading',waveFile)
        data, sr = sf.read(waveFile, dtype='float32')
        if sr != SAMPLERATE:
            data = data.T
            data = librosa.resample(data, sr, SAMPLERATE)
            concat_data = np.append(concat_data,data)
        else:
            concat_data = np.append(concat_data, data)

    return concat_data

def pre_process(samples):

    # preemphasis
    proc_samples = signal.lfilter( [1, EMPHASISCOEFF], [1], samples )

    return proc_samples
    
def post_process(samples):

    # deemphasis
    proc_samples = signal.lfilter( [1], [1, EMPHASISCOEFF], samples )

    return proc_samples

def scale_data(samples):
    samples = samples - samples.min()
    samples = samples / (samples.max() - samples.min())
    samples = (samples - 0.5) * 2
    return samples 

def main():
    print(f"AudioRNN starting at {time.ctime()}")
    script_dir = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--datadir", help="root directory of data", default="McGill")
    parser.add_argument("-m", "--modelfile", help="create audio from existing model file", default="AudioRNN.h5")
    parser.add_argument("-t", "--train", help="train the model", action="store_true")
    parser.add_argument("-a", "--audiofile", help="create audio file", default="output.wav")
    arg = parser.parse_args()

    modelfile = os.path.join(script_dir, arg.modelfile)
    audio_file = os.path.join(script_dir, arg.audiofile)
    data_dir = os.path.join(script_dir, arg.datadir)

    raw_data = load_data(data_dir)
    data = pre_process(raw_data[:32000])

    # encode data to 8-bits
    data = to_ulaw(data)
    # decode data from 8-bits to int16
    data = from_ulaw(data)
    
    data = post_process(data)

    # NOTE:test audio data is correct
    write_audio(data.astype('int16'), audio_file)

    print(f"AudioRNN completed at {time.ctime()}")

if __name__ == '__main__':
    main()
