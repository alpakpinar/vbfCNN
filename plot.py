#!/usr/bin/env python

import os
import sys
import re
import pickle
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import colors
from tqdm import tqdm

pjoin = os.path.join

class Plotter():
    def __init__(self, infile) -> None:
        '''Reads data from an input pkl file and plots a few events.'''
        self.infile = infile
        self._load_data()

        self.dataset_name = os.path.basename(self.infile).replace('.pkl','').replace('images_','')

    def _load_data(self):
        with open(self.infile, 'rb') as f:
            self.data = pickle.load(f)

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
            im = self.data[ievent]

            etabins = np.linspace(-5,5,im.shape[0])
            phibins = np.linspace(-np.pi,np.pi,im.shape[1])

            fig, ax = plt.subplots()
            cmap = ax.pcolormesh(etabins, phibins, im.T, norm=colors.LogNorm(vmin=1e-2,vmax=1e2))

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

    p.set_outdir('./plots')
    p.plot()

if __name__ == '__main__':
    main()