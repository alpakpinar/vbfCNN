#!/usr/bin/env python

import os
import sys
import uproot
import numpy as np

from matplotlib import pyplot as plt
from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from tensorflow.keras.utils import to_categorical

from train import NeuralNet, DataReader

pjoin = os.path.join

def load_test_data():
    '''Load data to test the predictions made by CNN.'''
    infile = 'scripts/output/2021-10-29_vbfhinv_26Oct21_nanov8/merged_2017.root'
    reader = DataReader(infile, treename='Events')
    imarr, labelarr = reader.read_images_and_labels()
    _, X_test, _, Y_test = train_test_split(
            imarr, 
            labelarr, 
            test_size=0.33, 
            random_state=42
            )

    return X_test, to_categorical(Y_test)

def plot_confusion_matrix(model, Y_test, Y_pred, outdir, normalize='true'):
    '''Wrapper function to compute and plot confusion matrix based on the predictions given as input.'''
    cm = confusion_matrix(Y_test.argmax(axis=1), Y_pred.argmax(axis=1), normalize=normalize)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
    )
    fig, ax = plt.subplots()
    disp.plot(ax=ax)
    outpath = pjoin(outdir, 'confusion_matrix.pdf')
    fig.savefig(outpath)

def save_to_root(rootfile_to_update, Y_pred):
    '''Save a ROOT file with test events, including the predictions made by the model.'''
    tree = rootfile_to_update['Events']
    tree['PredictionLabels'] = Y_pred

def main():
    # Input h5 file where the weights are stored
    infile = sys.argv[1]
    
    outtag = os.path.basename(os.path.dirname(infile))
    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Rebuild the model and load it's weights from previous training
    nn = NeuralNet(
        num_filters=8,
        filter_size=3,
        pool_size=2
    )
    nn.build_model()
    nn.compile()

    model = nn.model
    model.load_weights(infile)

    X_test, Y_test = load_test_data()
    
    # Make predictions and plot confusion matrix
    Y_pred = model.predict(X_test)
    plot_confusion_matrix(model, Y_test, Y_pred, outdir=outdir)

    loss, acc = model.evaluate(X_test, Y_test)
    print(f'Test accuracy: {acc*100:.3f}%')

    # TODO:
    # Save the predictions to a ROOT file
    predictions = Y_pred.argmax(axis=1)

    # ROOT file to update with predictions
    # rootdir = f'./scripts/output/{outtag}'
    # rootpath = pjoin(rootdir, 'merged.root')
    # with uproot.update(rootpath) as f:
        # save_to_root(f, predictions)


if __name__ == '__main__':
    main()