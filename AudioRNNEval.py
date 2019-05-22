import matplotlib.pyplot as plt
import pickle

dataFile = 'AudioRNN.npy'
data = pickle.load(open(dataFile, "rb"))

plt.figure('AudioRNN loss')
plt.plot(data['loss'])
plt.plot(data['val_loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
#plt.yscale('log')
plt.grid(True)

plt.figure('AudioRNN Accuracy')
plt.plot(data['acc'])
plt.plot(data['val_acc'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
#plt.yscale('log')
plt.grid(True)

plt.show()
