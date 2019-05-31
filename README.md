# AudioRNN

  A recurrent neural audio generation alogrithm that trains on and predicts PCM samples.
  
  Requirements(Pipfile included for creating virtual environment with all requirements):

- keras
- numpy
- soundfile
- scipy
- matplotlib
    
# Introduction

  Neural audio generation alogrithms require generating large datasets based on the long time context needed to accurately predict subsequent samples. Training to predict time series data with audio is a highly computationally complex problem due to the large sampling rates of high quality audio. Accuray is also a problem since the output range of PCM samples is (-32768, 32767). So AudioRNN predicts a probability distribution over the values 0-255 which is treated as a μ-law encoded LPCM sample, which is then transformed into a PCM sample with post-processing. This allows the model to train categorically instead of a more complex regresssion problem.
  AudioRNN is a highly parameterized model that allows for the loading of data at any sample rate and converting it to a specified training sample rate. It also allows for generating datasets and training with a user specified time context. All input and targets are quantized to 8-bit representations by way of the μ-law encoding. Since the μ-law encoding is susceptible to noise in the higher frequencies, a pre-emphasis IR filter is applied to the audio prior to the encoding to increase the SNR at higher frequencies. The output of the model is decoded back to PCM and passed through the inverse of the pre-emphasis filter to undo the processing done on the input. 
  
# Scripts
- AudioRNN.py
  Master script used for generating data, training the model and inference (generating audio).
  
- AudioRNNData.py
  Helper script with audio dataset generation functions, can also be run stand alone.

- AudioRNNEval.py
  Evaluation script for plotting a trained model's loss and accuracy.
  
# GRU Network

![alt text](https://github.com/mpc6/AudioRNN/blob/master/GRU.png "GRU network")

# Model Metrics

# Sample Audio 

  - [Audio Sample with 1000@8kHz sample time context](https://mpc6.github.io/AudioRNNDemo/output-1000ts.wav)
  - [Audio Sample with 128@8kHz sample time context](https://mpc6.github.io/AudioRNNDemo/output-128ts.wav)
  
# Future Work

