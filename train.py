#!/usr/bin/env python

import os
import sys
import uproot
import numpy as np

from matplotlib import pyplot as plt
from pprint import pprint

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical

pjoin = os.path.join

class DataReader():
    def __init__(self, infile, treename='Events') -> None:
        '''
        Per dataset data reader.
        Will accept an input ROOT file and read the jet or event images for all events.
        >> read_image(<imageName>) method can be used to extract an array of 2D event images.
        '''
        self.infile = infile
        self.treename = treename
    
    def _get_dimensions(self, tree, imname):
        # The name of the table to look for the eta/phi shapes
        if 'EventImage' in imname:
            tableName = 'EventImage'
        elif 'JetImage' in imname:
            tableName = 'JetImage'
        else:
            raise RuntimeError(f'Invalid image name: {imname}')

        nEtaBins = int(tree[f'{tableName}_nEtaBins'].array()[0])
        nPhiBins = int(tree[f'{tableName}_nPhiBins'].array()[0])

        return nEtaBins, nPhiBins

    def read_images_and_labels(self, imname='EventImage_pixelsAfterPUPPI', labelname='DatasetLabel', normalize=True):
        '''
        Returns a MxN NumPy array containing event images together with an Mx1 array containing event labels.
        M: Number of events
        N: Size of each image
        '''
        with uproot.open(self.infile) as f:
            t = f[self.treename]
            imarr = t[imname].array().to_numpy()
            labelarr = t[labelname].array().to_numpy()

            # Normalize
            if normalize:
                imarr = imarr.astype(np.float16) / 255.

            # Reshape each image into 2D!
            nEtaBins, nPhiBins = self._get_dimensions(t, imname)
            # Reshape: Shape should also include the depth of the data, which is 1 in our case
            new_shape = imarr.shape[0], nEtaBins, nPhiBins, 1
            imarr = imarr.reshape(new_shape)

        return imarr, labelarr

class NeuralNet():
    def __init__(self, num_filters, filter_size, pool_size, input_shape=(40,20,1)) -> None:
        '''Wrapper class for Keras Neural Network implementation.'''
        # Set some hyperparameters for our CNN
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.pool_size = pool_size
        
        self.input_shape = input_shape

    def build_model(self):
        self.model = Sequential([
            Conv2D(self.num_filters, self.filter_size, input_shape=self.input_shape),
            Conv2D(self.num_filters, self.filter_size),
            MaxPooling2D(pool_size=self.pool_size),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(3, activation='softmax') # 0: VBF H(inv), 1: EWK V+jets, 2: QCD V+jets
        ])
    
    def compile(self):
        self.model.compile(
            'adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )

    def train(self, X_train, Y_train, X_test, Y_test, epochs=20):
        history = self.model.fit(
            X_train,
            to_categorical(Y_train),
            epochs=epochs,
            validation_data=(X_test, to_categorical(Y_test)),
        )
        return history

    def save_weights(self, outfile):
        self.model.save_weights(outfile)

    def predict(self, testdata):
        return np.argmax(self.model.predict(testdata))

def plot_accuracy(history, outdir):
    '''Plot training and validation accuracies as a function of number of epochs.'''
    fig, ax = plt.subplots()
    training_acc = history.history['accuracy']
    validation_acc = history.history['val_accuracy']
    ax.plot(training_acc, label='Training')
    ax.plot(validation_acc, label='Validation')
    
    ax.set_xlabel('Epoch #')
    ax.set_ylabel('Training Accuracy')
    ax.legend()

    outpath = pjoin(outdir, 'training_acc.pdf')
    fig.savefig(outpath)
    plt.close(fig)

def main():
    # Path to the merged (via hadd) ROOT file
    infile = sys.argv[1]

    tag = os.path.basename(os.path.dirname(infile))
    # Directory to save output
    outdir = f'./output/{tag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Read image and label data
    reader = DataReader(infile, treename='Events')
    imarr, labelarr = reader.read_images_and_labels()

    # Split the data into training and testing sections
    X_train, X_test, Y_train, Y_test = train_test_split(
            imarr, 
            labelarr, 
            test_size=0.33, 
            random_state=42
            )

    # Construct the neural net and train
    nn = NeuralNet(
        num_filters=8,
        filter_size=3,
        pool_size=2
    )
    
    nn.build_model()
    nn.compile()

    history = nn.train(
        X_train, Y_train, 
        X_test, Y_test,
        epochs=10
    )

    plot_accuracy(history, outdir)

    # Output file to save weights
    outfile = pjoin(outdir, 'weights.h5')
    nn.save_weights(outfile)

if __name__ == '__main__':
    main()