# AudioRNN

  A recurrent neural audio generation alogrithm that trains on and predicts PCM samples.
  
  Requirements:

- keras
- numpy
- soundfile
- scipy
- matplotlib
    
# Introduction

  Neural audio generation alogrithms require generating large datasets based on the long time context needed to accurately predict subsequent samples. Training to predict time series data with audio is a highly computationally complex problem due to the large sampling rates of high quality audio. Accuray is also a problem to the output domain of PCM samples (-32768, 32767). AudioRNN is a highly parameterized model that allows for the loading of data at any sample rate and converting it to a specified training sample rate. It also reduces the complexity of inference by training on and predicting Linear Pulse Code Modulation. All input and targets are quantized to 8-bit representations by way of the \mu -law encoding.
