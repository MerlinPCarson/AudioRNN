import h5py
import os, sys
import numpy as np
from keras.utils import to_categorical

def dataset_generator(data, time_steps, chunk_size):
    num_chunks = 1
    X = []
    Y = []
    for frame_start in range(len(data)-time_steps - 1):
        # get frame and normalize
        frame = data[frame_start:frame_start+time_steps]
        # append frame of data to dataset 
        X.append(frame.reshape(time_steps,1))
        # get ulaw encoded sample after frame for target
        Y.append(data[frame_start+time_steps])
        if len(X) == chunk_size:
            print(f"Yielding chunk {num_chunks}, % done {frame_start/(len(data)-time_steps -1)}")
            yield np.array(X, dtype='float32'), np.array(Y, dtype='uint8').reshape(len(Y), 1)
            X = []
            Y = []
            num_chunks += 1


    print(f"Yielding final chunk {num_chunks}, % done {frame_start/(len(data)-time_steps -1)}")
    yield np.array(X, dtype='float32'), np.array(Y, dtype='uint8').reshape(len(Y), 1)

def load_audio_from_HDF5(data_file):
    print("Loading data from HDF5 file.")
    with h5py.File(data_file, 'r') as hf:
        return np.array(hf['AudioRNNdata'])

def save_data_to_HDF5(data, time_steps, outfile):
    chunk_size = len(data)//100
    print("Saving data to HDF5 file.")
    audio_gen = dataset_generator(data, time_steps, chunk_size)
    with h5py.File(outfile, 'w') as hf:
        x_train, y_train = next(audio_gen)
        print(x_train.shape, y_train.shape)
        hf.create_dataset('x_train', data = x_train, chunks=True, maxshape=(None, 1000, 1))
        hf.create_dataset('y_train', data = y_train, chunks=True, maxshape=(None, 1))
        while True:
            x_train, y_train = next(audio_gen)
            hf["x_train"].resize((hf["x_train"].shape[0] + x_train.shape[0]), axis = 0)
            hf["x_train"][-x_train.shape[0]:] = x_train
    
            hf["y_train"].resize((hf["y_train"].shape[0] + y_train.shape[0]), axis = 0)
            hf["y_train"][-y_train.shape[0]:] = y_train

            hf.flush()

            if(x_train.shape[0] <  chunk_size):
                break 

def main():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    input_file = os.path.join(script_dir, 'AudioData.h5')
    output_file = os.path.join(script_dir, 'AudioRNNData.h5')
    time_steps = 1000

    data = load_audio_from_HDF5(input_file)

    save_data_to_HDF5(data, time_steps, output_file)

if __name__ == '__main__':
    main()
    
