import numpy as np
import soundfile as sf
import librosa
import argparse
import pickle
import glob, os, time, math
from scipy import signal
from audioop import lin2ulaw, ulaw2lin

# Keras packages model
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, CuDNNGRU
from keras.callbacks import ModelCheckpoint #, EarlyStopping, CoolDown
from keras.optimizers import Adam
from keras.utils import to_categorical

SAMPLERATE = 16000
EMPHASISCOEFF = -0.95

TO_ULAW_SCALE = 255.0/32767.0
FROM_ULAW_SCALE = TO_ULAW_SCALE ** -1

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
    print('Pre-Processing data')

    # preemphasis
    proc_samples = signal.lfilter( [1, EMPHASISCOEFF], [1], samples )

    return proc_samples
    
def post_process(samples):

    # deemphasis
    proc_samples = signal.lfilter( [1], [1, EMPHASISCOEFF], samples )

    return proc_samples

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

def scale_data(samples):
    samples = samples - samples.min()
    samples = samples / (samples.max() - samples.min())
    samples = (samples - 0.5) * 2
    return samples 

def model(batch_size, time_steps, num_neurons):
    x_in = Input(batch_shape=(batch_size, time_steps, 1))
    x = Dense(num_neurons, activation='relu')(x_in)
    x = CuDNNGRU(num_neurons, return_sequences=False, stateful=False)(x)
    x = Dense(256, activation='softmax')(x)
    model = Model(inputs=[x_in], outputs=[x])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    return model

def create_dataset(data, time_steps):
    print('Creating Dataset')
    X = []
    Y = []
    for x in range(len(data)//time_steps - 1):
        # get frame and normalize
        frame = data[x:x+time_steps]/255
        # append frame of data to dataset 
        X.append(frame.reshape(time_steps,1))
        # get ulaw encoded sample after frame for target
        Y.append(data[x+time_steps])

    return np.array(X, dtype='float32'), to_categorical(Y, num_classes=256)

def main():
    print(f"AudioRNN starting at {time.ctime()}")
    script_dir = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--datadir", help="root directory of data", default="McGill")
    parser.add_argument("-m", "--modelfile", help="create audio from existing model file", default="AudioRNN.h5")
    parser.add_argument("-t", "--train", help="train the model", action="store_true")
    parser.add_argument("-e", "--epochs", help="Number of epochs", default=2)
    parser.add_argument("-b", "--batchsize", help="Number of batches per epoch", default=1)
    parser.add_argument("-ts", "--timesteps", help="Number of samples in time context", default=1000)
    parser.add_argument("-n", "--neurons", help="Number of neurons per layer", default=256)
    parser.add_argument("-a", "--audiofile", help="create audio file", default="output.wav")
    arg = parser.parse_args()

    # file arguments
    model_file = os.path.join(script_dir, arg.modelfile)
    audio_file = os.path.join(script_dir, arg.audiofile)
    data_dir = os.path.join(script_dir, arg.datadir)

    # model arguments
    epochs = arg.epochs
    batch_size = arg.batchsize
    time_steps = arg.timesteps
    neurons_per_layer = arg.neurons

    # load data
    print("[Data Preperation]")
    raw_data = load_data(data_dir)
    print(f'Number of samples: {len(raw_data)} Length of data: {len(raw_data)/SAMPLERATE} secs')
    data = pre_process(raw_data)
    
    # encode data to 8-bits
    data = to_ulaw(data)

    x_train, y_train = create_dataset(data, time_steps)
    print(f"Shape of input data: {x_train.shape} Shape of target data: {y_train.shape}")

    # build model
    AudioRNN = model(batch_size, time_steps, neurons_per_layer)
    AudioRNN.summary()
    
    print('[Initiating training]')
    best_model_checkpoint = ModelCheckpoint(model_file, save_best_only=True) 
    model_history = AudioRNN.fit(x_train, y_train, validation_split=0.1, batch_size=batch_size, epochs=epochs, callbacks=[best_model_checkpoint])
    pickle.dump(model_history.history, open(model_file.split('.')[0] + '.npy', "wb"))

    # decode data from 8-bits to int16
    #data = from_ulaw(data)
    
    #data = post_process(data)

    # NOTE:test audio data is correct
    #write_audio(data.astype('int16'), audio_file)

    print(f"AudioRNN completed at {time.ctime()}")

if __name__ == '__main__':
    main()
