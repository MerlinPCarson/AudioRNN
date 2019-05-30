# AudioRNN

  A recurrent neural audio generation alogrithm that trains on and predicts PCM samples.
  
  Requirements:

- keras
- numpy
- soundfile
- scipy
- matplotlib
    
# Introduction

  Neural audio generation alogrithms require generating large datasets based on the long time context needed to accurately predict subsequent samples. Training to predict time series data with audio is a highly computationally complex problem due to the large sampling rates of high quality audio. Accuray is also a problem to the output domain of PCM samples (-32768, 32767). AudioRNN is a highly parameterized model that allows for the loading of data at any sample rate and converting it to a specified training sample rate. It also reduces the complexity of inference by training on and predicting Linear Pulse Code Modulation. All input and targets are quantized to 8-bit representations by way of the μ-law encoding. Since the μ-law encoding is susceptible to noise in the higher frequencies, a pre-emphasis IR filter is applied to the input to the model to increase the SNR at higher frequencies. The output of the model is passed through the inverse of the pre-emphasis filter to undo the processing done on the input. 
  
# Scripts
- AudioRNN.py
  Master script used for generating data, training the model and inference (generating audio).
  
- AudioRNNData.py
  Helper script with audio dataset generation functions, can also be run stand alone.

- AudioRNNEval.py
  Evaluation script for plotting a trained model's loss and accuracy.
  
# GRU Network

![alt text](https://github.com/mpc6/AudioRNN/blob/master/GRU.png "GRU network")

# Sample Audio (not from AudioRNN)
[Sample of AudioRNN](https://github.com/mpc6/AudioRNN/blob/master/100981__mo-damage__atari-speech.wav)

