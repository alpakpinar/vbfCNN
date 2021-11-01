# VBF CNN

Keras-based convolutional neural network (CNN) implementation for classifying VBF images vs. V+jets backgrounds.

## Setup on LPC

If you want to run this code at LPC infrastructure, use these instructions to create and activate a working environment:

```
ENVNAME="vbfCNNenv" # Name you want to give to the virtual environment
python -m venv ${ENVNAME}
source ${ENVNAME}/bin/activate
```

You can leave the environment by typing `deactivate`.

### Package Installation
You can fetch the code from the GitHub repository and install dependencies as follows:

```
git clone git@github.com:alpakpinar/vbfCNN.git
cd vbfCNN/
pip install -r requirements.txt
```

## Running The Code

Once the setup is complete, one can run the full training and testing code in a couple of steps.

### Pre-processing

First, the input pickle files (in a compressed format) need to be converted to ROOT files before the training could be run.
To achieve this, one can use the `to_tree.py` command, located under `scripts` directory:

```
./to_tree.py <path_to_dir> # Point the script to the directory containing pkl.gz files
```

The script will produce per dataset ROOT files under `scripts/output`. These files are then can be merged via the use of `hadd`:

```
cd /path/to/dir/withROOTFiles
hadd merged.root *.root
```

### Training and Testing

Using the merged ROOT file created in the earlier step, one can train the neural network. To do that, run:

```
./train.py /path/to/merged.root
```

`train.py` script supports a number of additional command line options:

- `--learningrate`: Learning rate for the Adam optimizer, defaults to `0.01`
- `--testsize` : Fractional size of the testing dataset, defaults to `0.33` 
- `--numepochs` : Number of epochs to train the neural network, defaults to `50`

`train.py` will save the weights of the trained network as a `weights.h5` file under the `output` directory, together with plots of loss and accuracy as a function of epoch number. Using the stored weights, one can run the testing code:

```
./predict.py /path/to/weights.h5
```

This script will report the testing accuracy of the given model, and will plot a confusion matrix, saved under the `output` directory.