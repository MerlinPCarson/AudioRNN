import numpy as np
import soundfile as sf
import librosa
import argparse
import pickle
import h5py
from tqdm import tqdm
import glob, os, time, math
from scipy import signal
import AudioRNNData as DataGen
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#from audioop import lin2ulaw, ulaw2lin

# Keras packages model
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, CuDNNGRU, CuDNNLSTM, Dropout, BatchNormalization, concatenate
from keras.callbacks import Callback, ModelCheckpoint, CSVLogger #, EarlyStopping, CoolDown
from keras.optimizers import Adam
from keras.utils import to_categorical


#SAMPLERATE = 16000
EMPHASISCOEFF = 0.85

#TO_ULAW_SCALE = 255.0/32767.0
#FROM_ULAW_SCALE = TO_ULAW_SCALE ** -1

class SaveAudioCallback(Callback):
    def __init__(self, ckpt_freq, gen_length, sample_rate, time_steps, audio_context, batch_size):
        super(SaveAudioCallback, self).__init__()
        self.ckpt_freq = ckpt_freq
        self.audio_context = audio_context 
        self.gen_length = gen_length
        self.sample_rate = sample_rate
        self.time_steps = time_steps
        self.audio_context = audio_context
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs={}):
        if (epoch+1)%self.ckpt_freq==0:
            ts = str(int(time.time()))
            audio_file = os.path.join('output/', 'ckpt_'+ts+'.wav')
            audio = generate_audio(self.model, self.gen_length, self.sample_rate, self.time_steps, self.audio_context, self.batch_size)
            write_audio(post_process(audio).astype('int16'), audio_file, self.sample_rate)


def write_audio(samples, file_name, sample_rate):

    sf.write(file_name, samples, sample_rate, subtype='PCM_16')
    print('Audio saved to disk.')

def load_data(datadir, sample_rate):

    concat_data = np.array([],dtype='float32')

    for waveFile in glob.glob(os.path.join(datadir, '**', '*.wav'),recursive=True):
        print('loading',waveFile)
        data, sr = sf.read(waveFile, dtype='float32')
        if sr != sample_rate:
            data = data.T
            data = librosa.resample(data, sr, sample_rate)
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

def standardize(data):
   return (data - data.min())/(data.max() - data.min()) 

def from_ulaw(samples):
    dec_samples = []
    for sample in tqdm(samples):
        ampl_val_8 = ((((sample) / 255.0) - 0.5) * 2.0)
        ampl_val_16 = (np.sign(ampl_val_8) * (1/256.0) * ((1 + 256.0)**abs(ampl_val_8) - 1)) * 2**15
        dec_samples.append(ampl_val_16)
    return np.array(dec_samples, dtype=np.float)

def to_ulaw(samples):
    enc_samples = [int((np.sign(sample) * (np.log(1 + 256*abs(sample)) / (
            np.log(1+256))) + 1)/2.0 * 255) for sample in tqdm(samples)]
    return np.array(enc_samples, dtype=np.uint8)

def scale_data(samples):
    samples = samples - samples.min()
    samples = samples / (samples.max() - samples.min())
    samples = (samples - 0.5) * 2
    return samples 

def model(batch_size, time_steps, num_neurons):
    x_in = Input(batch_shape=(batch_size, time_steps, 1))
    x = Dense(num_neurons, activation='relu')(x_in)
    x = BatchNormalization()(x)
    rnn_in1 = concatenate([x_in, x])
    x = CuDNNGRU(num_neurons, return_sequences=True, stateful=False)(rnn_in1)
    x = BatchNormalization()(x)
    rnn_in2 = concatenate([x_in, x])
    #x = Dropout(0.4)(x)
    x = CuDNNGRU(num_neurons, return_sequences=False, stateful=False)(rnn_in2)
    x = BatchNormalization()(x)
    #x = Dropout(0.4)(x)
    x = Dense(256, activation='softmax')(x)
    model = Model(inputs=[x_in], outputs=[x])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

def create_dataset(data, time_steps, time_shift):
    print('Creating Dataset')
    X = []
    Y = []
    for frame_start in range(0, len(data)-time_steps - 1, time_shift):
        # get frame and normalize
        frame = data[frame_start:frame_start+time_steps]
        # append frame of data to dataset 
        X.append(frame.reshape(time_steps,1))
        # get ulaw encoded sample after frame for target
        Y.append(data[frame_start+time_steps])

    return np.array(X, dtype='float32'), to_categorical(Y, num_classes=256)

def load_from_HDF5(data_file, num_examples, start_example=0):
    print("Loading data from HDF5 file.")
    with h5py.File(data_file, 'r') as hf:
        x_train = hf['x_train'][start_example:start_example+num_examples]
        y_train = hf['y_train'][start_example:start_example+num_examples]

    return x_train.astype('float32'), y_train.astype('uint8')

def load_audio_from_HDF5(data_file):
    print("Loading audio data from HDF5 file.")
    with h5py.File(data_file, 'r') as hf:
        return np.array(hf['AudioRNNData'])

def save_audio_to_HDF5(data, data_file):
    print("Saving audio data to HDF5 file.")
    with h5py.File(data_file, 'w') as hf:
        hf.create_dataset('AudioRNNData', data=data)

def generate_audio(AudioRNN, gen_length, sample_rate, time_steps, audio_prompt, batch_size):
    audio = []
    audio_prompt /= 255     # normalize data
    print(f"Generating {gen_length} secs of audio.")
    for sample in tqdm(range(int(gen_length*sample_rate))):
        output = AudioRNN.predict(audio_prompt.reshape(batch_size,time_steps,1))
        pred_sample = np.argmax(output)
        audio.append(pred_sample)
        pred_sample /= 255
        audio_prompt = np.append(audio_prompt, pred_sample)
        audio_prompt = audio_prompt[1:]

    #print(audio)
    return from_ulaw(audio) 
    

def main():
    print(f"AudioRNN starting at {time.ctime()}")
    script_dir = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--datadir", help="root directory of data", default="McGillSmall16k")
    parser.add_argument("-sr", "--samplerate", help="audio sample rate", type=int, default=8000)
    parser.add_argument("-df", "--datafile", help="HDF5 file to save data to", default="AudioRNNData.h5")
    parser.add_argument("-af", "--audiodatafile", help="HDF5 file to save PCM data to", default="AudioData.h5")
    parser.add_argument("-m", "--modelfile", help="create audio from existing model file", default="AudioRNN.h5")
    parser.add_argument("-t", "--train", help="train the model", action="store_true")
    parser.add_argument("-g", "--generate", help="generate_audio", action="store_true")
    parser.add_argument("-gl", "--genlength", help="length of generate_audio in secs", type=float, default=1)
    parser.add_argument("-sh5", "--saveHDF5", help="save preprocessed data to HDF5 file", action="store_true")
    parser.add_argument("-lh5", "--loadHDF5", help="load preprocessed data from HDF5 file", action="store_true")
    parser.add_argument("-lah5", "--loadaudioHDF5", help="load preprocessed data from HDF5 file", action="store_true")
    parser.add_argument("-sah5", "--saveaudioHDF5", help="save preprocessed data to HDF5 file", action="store_true")
    parser.add_argument("-e", "--epochs", help="Number of epochs", type=int, default=100)
    parser.add_argument("-b", "--batchsize", help="Number of batches per epoch", type=int, default=128)
    parser.add_argument("-ne", "--numexamples", help="Number of examples to use from dataset", type=int, default=5000)
    parser.add_argument("-ts", "--timesteps", help="Number of samples in time context", type=int, default=1000)
    parser.add_argument("-tsft", "--timeshift", help="Number of samples to skip for each example", type=int, default=1000)
    parser.add_argument("-n", "--neurons", help="Number of neurons per layer", type=int, default=256)
    parser.add_argument("-a", "--audiofile", help="create audio file", default="output.wav")
    arg = parser.parse_args()

    # training arguments
    train = arg.train 

    # generate audio arguments
    gen_audio = arg.generate 
    gen_length = arg.genlength
    num_examples = arg.numexamples

    # data arguments
    load_HDF5 = arg.loadHDF5 
    save_HDF5 = arg.saveHDF5 
    load_audio_HDF5 = arg.loadaudioHDF5
    save_audio_HDF5 = arg.saveaudioHDF5
    sample_rate = arg.samplerate

    # model arguments
    epochs = arg.epochs
    batch_size = arg.batchsize
    time_steps = arg.timesteps
    time_shift = arg.timeshift
    neurons_per_layer = arg.neurons

# NOTE: for debugging
    train = True
    load_HDF5 = True 
    #gen_audio = True
    #save_HDF5 = True
    #load_audio_HDF5 = True
    #time_steps = 128
    arg.datadir = 'Opeth' 
    #save_audio_HDF5 = True


    # file arguments
    model_file = os.path.join(script_dir, arg.modelfile)
    audio_file = os.path.join(script_dir, arg.audiofile)
    data_dir = os.path.join(script_dir, arg.datadir)
    data_file = os.path.join(script_dir, arg.datafile.replace('.h5', str(time_steps) + '.h5'))
    audio_data_file = os.path.join(script_dir, arg.audiodatafile.replace('.h5', str(sample_rate) + '.h5'))

    # load audio data, if dataset is not loaded from HDF5
    if not load_HDF5 and train:
        if load_audio_HDF5:
            print(f"loading audio data from {audio_data_file}")
            data = load_audio_from_HDF5(os.path.join(script_dir, audio_data_file))
            #data = data[:478935]
            #data = from_ulaw(data)
            #data = post_process(data)
            #write_audio(data.astype('int16'), audio_file, sample_rate)     # test HDF5 data loader/ mu-law transform
            print(f"Data max: {data.max()}, Data min: {data.min()}")
        else:
            print("[Data Preperation]")
            print(f"loading waves from {data_dir}")
            raw_data = load_data(data_dir, sample_rate)
            #write_audio(raw_data, audio_file, sample_rate)     # test data loader/resampler
            print(f'Number of samples: {len(raw_data)} Length of data: {len(raw_data)/sample_rate} secs'"""  """)
            #raw_data = raw_data[:24000]
            #data = pre_process(raw_data)
    
            # encode data to 8-bits
            print('Encoding data as mu-law')
            data = to_ulaw(raw_data)
            #data = from_ulaw(data)
            #data = post_process(data)
            #write_audio(data.astype('int16'), audio_file, sample_rate)     # test mu-law transform


        # write pre processed audio to HDF5 file
        if save_audio_HDF5:
            print(f"saving pre-processed audio data to {audio_data_file}")
            save_audio_to_HDF5(data, os.path.join(script_dir, audio_data_file))

        if save_HDF5:
            # create datasets and save to HDF5 file
            print(f"saving dataset to {data_file}")
            DataGen.save_data_to_HDF5(data, time_steps, data_file)

    # train the model
    if train:

        # load the datasets
        assert os.path.exists(data_file), f"Data file {data_file}, does not exists!"
        print(f"loading dataset from {data_file}")
        x_train, y_train = load_from_HDF5(data_file, num_examples)
       
        # normalize the data 
        x_train = x_train/255

        # get training and validation sets
        #x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.01, shuffle=False )
        # data must be a multiple of batch size
        num_train_samples = batch_size * (len(x_train)//batch_size)
        x_train = x_train[0:num_train_samples,:,:]
        y_train = y_train[0:num_train_samples,:]
        #num_valid_samples = batch_size * (len(x_valid)//batch_size)
        #x_valid = x_valid[0:num_valid_samples,:,:]
        #y_valid = y_valid[0:num_valid_samples,:]
        print(f"TRAINING: Shape of input data {x_train.shape}, Shape of target data {y_train.shape}")
        #print(f"VALIDATION: Shape of input data {x_valid.shape}, Shape of target data {y_valid.shape}")

        # build model
        AudioRNN = model(batch_size, time_steps, neurons_per_layer)
        AudioRNN.summary()

        print('[Initiating training]')
        csv_logger = CSVLogger('AudioRNN.log')
        best_valid_model_checkpoint = ModelCheckpoint(model_file, save_best_only=True) 
        best_train_model_checkpoint = ModelCheckpoint(model_file.split('.')[0] + '_train.h5', save_best_only=True, monitor='loss', mode='min') 
        #escb = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
        #audio_prompt = x_train[0:batch_size, :, :]
        #gen_audio_callback = SaveAudioCallback(1, 0.5, sample_rate, time_steps, audio_prompt, batch_size)
        model_history = AudioRNN.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=[csv_logger, best_train_model_checkpoint, best_valid_model_checkpoint])
        with open(model_file.split('.')[0] + '.npy', "wb") as outfile:
            pickle.dump(model_history.history, outfile)

    # generate audio from the trained model
    if gen_audio:
        AudioRNN = model(1, time_steps, neurons_per_layer)
        print(f"loading weights from {model_file}")
        AudioRNN.load_weights(model_file)
        AudioRNN.summary()
        example =  np.random.randint(0,num_examples)
        example = 0 
        print(f"Prompting audio generation with example {example}")
        audio_prompt, _ = load_from_HDF5(data_file, 1, start_example=example)
        audio = generate_audio(AudioRNN, gen_length, sample_rate, time_steps, audio_prompt, 1)
        #write_audio(post_process(audio).astype('int16'), audio_file, sample_rate)
        write_audio(audio.astype('int16'), audio_file, sample_rate)

    # NOTE:test audio data and transformations are correct
    #data = from_ulaw(data)
    #data = post_process(data)
    #write_audio(data.astype('int16'), audio_file)

    print(f"AudioRNN completed at {time.ctime()}")

if __name__ == '__main__':
    main()
