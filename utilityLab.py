import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import glob
import argparse

def get_args(parser=None):
    # Argument Parser
    if parser is None:
        parser = argparse.ArgumentParser(description='Prepare data for training')
    parser.add_argument('--inputText', default='alice_in_wonderland.txt', type=str, help='File for training and influence time')
    args = parser.parse_args()
    return args

class util:
  def __init__(self, parser=None):
    args = get_args(parser)
    self.inputData = args.inputText
  
  
  # function for data prepare
  def data_prepare(self,args):
    # load ascii text and covert to lowercase
    print(args.inputText)
    raw_text = open(args.inputText, 'r', encoding='utf-8').read()
    raw_text = raw_text.lower()
    # create mapping of unique chars to integers, and a reverse mapping
    chars = sorted(list(set(raw_text)))
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    int_to_char = dict((i, c) for i, c in enumerate(chars))
    # summarize the loaded data
    n_chars = len(raw_text)
    n_vocab = len(chars)
    print ("Total Characters: ", n_chars)
    print ("Total Vocab: ", n_vocab)
    
    # prepare the dataset of input to output pairs encoded as integers
    seq_length = 100
    dataX = []
    dataY = []
    for i in range(0, n_chars - seq_length, 1):
      seq_in = raw_text[i:i + seq_length]
      seq_out = raw_text[i + seq_length]
      dataX.append([char_to_int[char] for char in seq_in])
      dataY.append(char_to_int[seq_out])
    n_patterns = len(dataX)
    print ("Total Patterns: ", n_patterns)
    # reshape X to be [samples, time steps, features]
    X = np.reshape(dataX, (n_patterns, seq_length, 1))
    # normalize
    X = X / float(n_vocab)
    # one hot encode the output variable
    y = np_utils.to_categorical(dataY)
    return X, y, dataX, dataY, int_to_char, n_vocab

  def models(X,y):
    # define the LSTM model
    model = Sequential()
    model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model
   




  