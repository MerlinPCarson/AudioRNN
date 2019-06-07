[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/mpc6/AudioRNN/blob/master/LICENSE.txt)

# AudioRNN

  A recurrent neural audio generation alogrithm that trains on and predicts PCM samples.
  
  Requirements (Pipfile included for creating virtual environment with all requirements):

- keras
- numpy
- soundfile
- scipy
- matplotlib
    
# Introduction

Neural audio generation alogrithms require generating large datasets based on the long time context needed to accurately predict subsequent samples. Training to predict time series data with audio is a computationally complex problem due to the large sampling rates of high quality audio. Accuray is also a problem since the output range of PCM samples is [-32768, 32767]. So, AudioRNN predicts a probability distribution over the values [0, 255] which represent the range of μ-law encoded LPCM samples. The sample selected from the distribution is transformed into a PCM sample with post-processing. This allows the model to train categorically instead of as a more complex regresssion problem.
  
AudioRNN is a highly parameterized model that allows for the loading of data at any sample rate and converting it to a specified training sample rate. It also allows for generating datasets and training with a user specified time context. All input and targets are quantized to 8-bit representations by way of the μ-law encoding. Since the μ-law encoding is susceptible to noise in the higher frequencies, a pre-emphasis IR filter is applied to the audio prior to the encoding to increase the SNR at higher frequencies. The output of the model is decoded back to PCM and passed through the inverse of the pre-emphasis filter to undo the processing done on the input.
  
# Scripts
- AudioRNN.py
  Master script used for generating data, training the model and inference (generating audio).
  
- AudioRNNData.py
  Helper script with audio dataset generation functions, can also be run stand alone.

- AudioRNNEval.py
  Evaluation script for plotting a trained model's loss and accuracy.
  
# AudioRNN Model

![alt text](https://github.com/mpc6/AudioRNN/blob/master/ReadmeAssets/GRU-AudioRNN.png "AudioRNN model")

# Training Data

 - [McGill Telecommunications & Signal Processing Laboratory
Multimedia Signal Processing (60 minutes of speech)](http://www-mmsp.ece.mcgill.ca/Documents/Data)

# Model Metrics

Training on 60 seconds of audio, it can be seen in the following accuracy and training loss graphs that the model is indeed capable of learning to predict samples with over a 99% accuracy.

![alt text](https://github.com/mpc6/AudioRNN/blob/master/ReadmeAssets/AudioRNN1min.png "AudioRNN model trained on 60 secs data")

Training on 10 minutes of audio resulted in a significantly lower accuracy and higher loss, as can be seen in the graphs below. This would suggest that the model's capacity is too small for this amount of data. It can be seen between the training curve (blue) and the validation curve (orange) that the model did generalize well. However, increasing the size of the network would take significantly longer to train.

![alt text](https://github.com/mpc6/AudioRNN/blob/master/ReadmeAssets/AudioRNN60mins.png "AudioRNN model trained on 60 mins data")

# Sample Audio 

  Trained with a 1000 sample time context at an 8kHz sample rate, approximately 1/8th of a second, the model is capable of mimicing speech qualities. However, this requires preparing each sample of the training data to have the previous 1000 samples prepended to it. Therefore, 60 mins of audio would require a dataset that contains a total of 60000 mins (~ 110GBs). Because of memory and time constraints, I was only able to train with ~ 8 minutes of speech, thus long term dependencies were not well generalized and the model was only capable of mimicing small sections of speech like audio.
  
   ![alt text](https://github.com/mpc6/AudioRNN/blob/master/ReadmeAssets/AudioRNNwav1000ts.png "AudioRNN model 1000 time steps")
  - [Audio Sample with 1000@8kHz sample time context](https://mpc6.github.io/AudioRNNDemo/output-1000ts.wav)
  
  Trained with a 128 sample time context at an 8kHz sample rate, ~ 1.6% of a second, I was able to create a dataset and train on the entire 60 minutes of speech audio. However, this time context is too short to learn speech like qualities. The model did learn to generate audio with phase, but the samples do not sound in any way like speech.
  
  ![alt text](https://github.com/mpc6/AudioRNN/blob/master/ReadmeAssets/AudioRNNwav128ts.png "AudioRNN model 128 time steps")
  - [Audio Sample with 128@8kHz sample time context](https://mpc6.github.io/AudioRNNDemo/output-128ts.wav)
  
# Future Work

- [ ] Create Keras data generator to training on larger amounts of data
- [ ] Experiment with longer time contexts
- [ ] Train on music data sets


