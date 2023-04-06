![VAME](https://github.com/saadchtouki/VAME_CS_RG/blob/6630b427b0575dad2d03394b3f423e38ebc596b4/Images/Banniere_readme.jpg)

![workflow](https://github.com/saadchtouki/VAME_CS_RG/blob/6630b427b0575dad2d03394b3f423e38ebc596b4/Images/Banniere_readme2.jpg)

# The project in a Nutshell
The project consists of segmenting car journeys, traced with the help of sensors, into hierarchical driving situations. We are basing this on the work done on VAME (Variational animal motion embedding). [VAME](https://github.com/LINCellularNeuroscience/VAME.git) is a framework to cluster animal behavioral signals obtained from pose-estimation tools. It is a [PyTorch](https://pytorch.org/) based deep learning framework which leverages the power of recurrent neural networks (RNN) to model sequential data. In order to learn the underlying complex data distribution, an RNN is used in a variational autoencoder setting to extract the latent state of the animal (in our case the latent state of the car) in every step of the input time series.

![behavior](https://github.com/saadchtouki/VAME_CS_RG/blob/a1fd6e4cb1bf86f3fb6a3e006f9c029dd3882cfe/Images/Behavioral_video.gif)
![example](https://github.com/saadchtouki/VAME_CS_RG/blob/a1fd6e4cb1bf86f3fb6a3e006f9c029dd3882cfe/Images/Exemple_visu_dyna.jpg)

The workflow consists of 6 steps and we explain them in detail [here](https://github.com/saadchtouki/VAME_CS_RG/wiki/Project-Workflow).

## Installation
To get started we recommend using [Anaconda](https://www.anaconda.com/distribution/) with Python 3.6 or higher. 
In order to create a virtual enviroment to store all the dependencies necessary for VAME, you must use the VAME.yaml file supplied here, by simply openning the terminal, running `git clone https://github.com/saadchtouki/VAME_CS_RG.git`, then type `cd VAME_CS_RG` then run: `conda env create -f VAME.yaml`).

* Go to the locally cloned VAME directory and run `python setup.py install` in order to install VAME in your active conda environment.
* Install the current stable Pytorch release using the OS-dependent instructions from the [Pytorch website](https://pytorch.org/get-started/locally/). Currently, VAME is tested on PyTorch 1.5. (Note, if you use the conda file we supply, PyTorch is already installed and you don't need to do this step.)

## Getting Started
First, you should make sure that you have a GPU powerful enough to train deep learning networks. In our paper, we were using a single Nvidia GTX 1080 Ti GPU to train our network. A hardware guide can be found [here](https://timdettmers.com/2018/12/16/deep-learning-hardware-guide/). Once you have your hardware ready, you can use the visualization tools by following the [workflow guide](https://github.com/saadchtouki/VAME_CS_RG/wiki/Project-Workflow).


### Authors and Code Contributors
This work is our final year project at CentraleSupélec.

The development is based on the work done in [VAME](https://github.com/LINCellularNeuroscience/VAME.git) (Variational animal motion embedding).
