#!/usr/bin/env python

import os
import sys
import re
import uproot
import pickle
import numpy as np

from matplotlib import pyplot as plt
from tqdm import tqdm

pjoin = os.path.join

class DatasetLabeler():
    def __init__(self) -> None:
        pass
    def get_labels(self, datasetname, numevents):
        '''Get properly shaped dataset labels.'''
        # VBF H(inv)
        if re.match('VBF_HToInv.*M125.*', datasetname):
            return np.zeros(numevents)
        # EWK V+jets
        elif re.match('EWK.*', datasetname):
            return np.ones(numevents)
        # QCD V+jets
        elif re.match('(Z\dJetsToNuNu|WJetsToLNu)_Pt.*FXFX.*', datasetname):
            return 2 * np.ones(numevents)
        
        raise RuntimeError(f'Cannot get a label for dataset: {datasetname}')

class ImageReader():
    def __init__(self, infile, treename='pixels') -> None:
        '''
        Per dataset image reader.
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

    def read_image(self, imname='EventImage_pixelsAfterPUPPI'):
        '''
        Returns a MxN NumPy array containing event images.
        M: Number of events
        N: Size of each image
        '''
        with uproot.open(self.infile) as f:
            t = f[self.treename]
            arr = t[imname].array().to_numpy()

            # Reshape each image into 2D!
            nEtaBins, nPhiBins = self._get_dimensions(t, imname)
            new_shape = arr.shape[0], nEtaBins, nPhiBins
            arr = arr.reshape(new_shape)

        return arr

class PickleSaver():
    def __init__(self, outdir) -> None:
        '''Saves the training data and labels to a pickle file.'''
        self.outdir = outdir

    def save(self, imageset, labelset, datasetname):
        imagefilename = pjoin(self.outdir, f'images_{datasetname}.pkl')
        labelfilename = pjoin(self.outdir, f'labels_{datasetname}.pkl')
        
        with open(imagefilename, 'wb+') as f:
            pickle.dump(imageset, f)

        with open(labelfilename, 'wb+') as f:
            pickle.dump(labelset, f)

class Processor():
    def __init__(self, inputdir) -> None:
        '''Wrapper main class to execute the jobs for all ROOT files in a given input directory.'''
        # List of files that we'll process
        self.files = [pjoin(inputdir, f) for f in os.listdir(inputdir) if f.endswith('.root')]

    def run(self):
        for inpath in tqdm(self.files):
            reader = ImageReader(inpath)
            imageset = reader.read_image()
            numevents = imageset.shape[0]

            datasetname = os.path.basename(inpath).replace('.root','')
            labeler = DatasetLabeler()
            labelset = labeler.get_labels(datasetname, numevents)
            
            # Save the data for training/testing!
            saver = PickleSaver(outdir='./output')
            saver.save(
                imageset=imageset,
                labelset=labelset,
                datasetname=datasetname
            )

def main():
    inputdir = sys.argv[1]
    processor = Processor(inputdir)

    processor.run()

if __name__ == '__main__':
    main()