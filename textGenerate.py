# Load LSTM network and generate text
import sys
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
import argparse
import glob
from utilityLab import util

#load data from argument
parser = argparse.ArgumentParser()
parser.add_argument("--inputText", default="alice_in_wonderland.txt", help="Data source Definition")
args = parser.parse_args()

# Data preprocessing
X,y, dataX, dataY, int_to_char, n_vocab = util.data_prepare(args.inputText, args)


# define the LSTM model
model = util.models(X,y)

# load the network weights
filename = "weights-improvement-04-2.2806.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# pick a random seed
start = numpy.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print(pattern)
print ("Seed:")
print ("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
# generate characters
for i in range(100):
	x = numpy.reshape(pattern, (1, len(pattern), 1))
	x = x / float(n_vocab)
	prediction = model.predict(x, verbose=0)
	index = numpy.argmax(prediction)
	result = int_to_char[index]
	seq_in = [int_to_char[value] for value in pattern]
	sys.stdout.write(result)
	pattern.append(index)
	pattern = pattern[1:len(pattern)]
print ("\nDone.")

#
#if __name__ == "__main__":
#   main(sys.argv[1:])