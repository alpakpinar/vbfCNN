#!/usr/bin/env python

import os
import sys
import gzip
import re
import uproot
import pickle
import numpy as np

from pprint import pprint
from tqdm import tqdm

pjoin = os.path.join

def is_qcd_v(datasetname):
    return re.match('(Z\dJetsToNuNu|WJetsToLNu).*Pt.*FXFX.*', datasetname)

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
        elif is_qcd_v(datasetname):
            return 2 * np.ones(numevents)
        
        raise RuntimeError(f'Cannot get a label for dataset: {datasetname}')

def main():
    # Path to directory containing pkl files
    inpath = sys.argv[1]
    infiles = [pjoin(inpath, f) for f in os.listdir(inpath) if f.endswith('.pkl.gz')]

    outtag = inpath.split('/')[-2]

    labeler = DatasetLabeler()

    for infile in tqdm(infiles):
        # Decompress the file and read the pickled contents
        with gzip.open(infile, 'rb') as fin:
            data = pickle.load(fin)
    
        # Get dataset name
        dataset = os.path.basename(infile).replace('.pkl.gz','')

        # Scout branches
        inputs = data.keys()

        # Label the dataset according to it's dataset name
        numevents = len(data[list(inputs)[0]])

        # Prescale: Randomly, take every 100th entry for QCD V+jets
        if is_qcd_v(dataset):
            prescale = 100
            np.random.seed(0)
            numevents_after_scale = int(np.floor(numevents / prescale))
            choices = np.random.choice(numevents, numevents_after_scale, replace=False)
        # Otherwise, we take all
        else:
            choices = np.ones(numevents, dtype=bool)
        
        labels = labeler.get_labels(dataset, numevents)
        labels = labels[choices]

        if len(labels) == 0:
            continue

        outdir = f'./output/{outtag}'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        outrootfile = pjoin(outdir, f'{dataset}.root')
    
        # Save the tree
        with uproot.recreate(outrootfile) as f:
            tree_data = {}
            for pixelname in inputs:
                if 'nBins' in pixelname:
                    tree_data[pixelname] = np.stack(data[pixelname][choices]).astype(np.uint16) 
                else:
                    tree_data[pixelname] = np.stack(np.array(data[pixelname])[choices])
            
            tree_data['DatasetLabel'] = labels
            f['Events'] = tree_data
    
if __name__ == '__main__':
    main()