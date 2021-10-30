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
        elif re.match('(Z\dJetsToNuNu|WJetsToLNu).*Pt.*FXFX.*', datasetname):
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
        labels = labeler.get_labels(dataset, numevents)

        outdir = f'./output/{outtag}'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        outrootfile = pjoin(outdir, f'{dataset}.root')
    
        # Save the tree
        with uproot.recreate(outrootfile) as f:
            tree_data = {}
            for pixelname in inputs:
                if 'nBins' in pixelname:
                    tree_data[pixelname] = np.stack(data[pixelname]).astype(np.uint16) 
                else:
                    tree_data[pixelname] = np.stack(data[pixelname])
            
            tree_data['DatasetLabel'] = labels
            f['Events'] = tree_data
    
if __name__ == '__main__':
    main()