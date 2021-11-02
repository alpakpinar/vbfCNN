#!/usr/bin/env python

import os
import sys
import re
import gzip
import pickle
import warnings
import numpy as np
import matplotlib.cbook

from matplotlib import pyplot as plt
from matplotlib import colors
from tqdm import tqdm
from pprint import pprint

pjoin = os.path.join

# Lets ignore some depreciation warnings from Matplotlib for now
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

class Plotter():
    def __init__(self, infile) -> None:
        '''Reads data from an input pkl file and plots a few events.'''
        self.infile = infile
        self._load_data()

        self.dataset_name = os.path.basename(self.infile).replace('.pkl.gz','').replace('images_','')

    def _load_data(self):
        # Decompress the file and read the pickled contents
        with gzip.open(self.infile, 'rb') as fin:
            data = pickle.load(fin)

            self.pixels = data['EventImage_pixelsAfterPUPPI']
            self.numEtaBins = int(data['EventImage_nEtaBins'][0])
            self.numPhiBins = int(data['EventImage_nPhiBins'][0])

    def _get_dataset_label(self):
        if re.match('VBF_HToInv.*M125.*', self.dataset_name):
            return 'VBF H(inv)'
        elif re.match('EWK(Z|W).*', self.dataset_name):
            return 'EWK V+jets'
        return 'QCD V+jets'

    def set_outdir(self, outdir):
        self.outdir = outdir
        if not os.path.exists(outdir):
            os.makedirs(outdir)

    def plot(self, numevents=5):
        for ievent in tqdm(range(numevents)):
            im = self.pixels[ievent]

            im = np.reshape(im, (self.numEtaBins, self.numPhiBins))

            etabins = np.linspace(-5,5,im.shape[0])
            phibins = np.linspace(-np.pi,np.pi,im.shape[1])

            fig, ax = plt.subplots()
            cmap = ax.pcolormesh(etabins, phibins, im.T, norm=colors.LogNorm(vmin=1e-2,vmax=255))

            cb = fig.colorbar(cmap, ax=ax)
            cb.set_label('Energy (GeV)')

            ax.set_xlabel(r'PF Candidate $\eta$')
            ax.set_ylabel(r'PF Candidate $\phi$')

            ax.text(0,1,self._get_dataset_label(),
                fontsize=14,
                ha='left',
                va='bottom',
                transform=ax.transAxes
            )

            ax.text(1,1,f'ievent={ievent}',
                fontsize=14,
                ha='right',
                va='bottom',
                transform=ax.transAxes
            )

            outfilename = pjoin(self.outdir, f'{self.dataset_name}_ievent_{ievent}.pdf')
            fig.savefig(outfilename)
            plt.close(fig)

def main():
    # Path to input pkl file
    infile = sys.argv[1]
    p = Plotter(infile)

    outtag = os.path.basename(os.path.dirname(infile))

    p.set_outdir(f'./plots/{outtag}')
    p.plot()

if __name__ == '__main__':
    main()