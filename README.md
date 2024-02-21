Linked within the repository is also the AltimeterAutoencoder repository, which contains its own documentation.

# Requirements 

The required packages are all included in the *grid* conda environment used in the grid interpolation. The environment.yml file may be found in the AltimeterGridding repository.

In order to utilize the program in this repository, a data set must be provided. The idea is to use the grid data set produced using the AltimeterGridding pipeline, however, both MEaSUREs and CMEMS data should be usable. The 5 day gap between gridded days in MEaSUREs may provide issues, which are not accounted for.

# Modules

Several methods of forecasting have been implemented and investigated, with varying degrees of complexity and necessary preparation. The models are implemented in [train_ConvAttention.py](train_ConvAttention.py), [train_AutoConvAttention.py](train_AutoConvAttention.py) and [train_AutoConvAttentionLatent.py](train_AutoConvAttentionLatent.py). On the HPC these may be executed as batch jobs using the [submit.sh](submit.sh) file, which is the job file, using:
```sh
$ bsub < submit.sh
```
and check the running jobs using:
```sh
$ bstat (-M) (-C)
```
Check the availability of nodes using:
```sh
$ nodestat hpc
```
and the length of the queue with:
```sh
$ classstat hpc
```

The different models may also be executed using the [train_test.ipynb](Models/AttentionConvLSTMAutoEncoder/train_test.ipynb) Jupyter Notebooks for a better overview of the running code. These are found in each of the subfolders of the *Models* folder where the auxillary functions for each implementation are also found.

The program is configured to be executed using a single combined netcdf file containing the entire grid data set. This may have to be changed for larger data sets, as compiling and storing these quickly becomes infeasible with increased resolution. This is the case for all three model implementations.

## ConvAttention

Most basic implementation of the Convolutional LSTM network with Attention. 

## AutoConvAttention

Added around the ConvAttention implementation is an Encoder-Decoder architecture. The Autoencoder is meant to near-perfectly de- and reconstruct the data representing it in a smaller-dimensional latent space speeding up the learning process.

## AutoConvAttentionLatent

The model architecture here is basically the same as that of AutoConvAttention. However, the input to the model is the latent space of the AltimeterAutoencoder model. This results in a nested latent space decreasing the number of parameters heavily. This speeds up training and decreases the feature space, which has shown to improve long term forecasts.


