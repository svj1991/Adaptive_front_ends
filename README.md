# End-to-end Source Separation using Adaptive Front-ends
End-to-end Source Separation using Adaptive Front-ends code. 

### Software requirements: 
* Python v > 3.5
* Numpy v > 1.12 
* Pytorch v > 0.1 
* Visdom

### Links
The paper is available [here](https://arxiv.org/pdf/1705.02514.pdf)

Separation examples are available [here](http://www.vshrikant.com/nn_adaptive_transforms.html)

### Commands
Run the following commands in the terminal for running the code:

python3 tsep.py -f 1024 -h 16 (separation with sdr cost, 1024 bases and hop 16)

python3 tsep.py -f 1024 -h 16 -mse (separation with mean squared error cost, 1024 bases and hop 16)

python3 tsep.py -f 1024 -h 16 -dft (separation with sdr cost, 1024 bases and hop 16, fourier bases without adaptive front-end)

### Brief Description

Source separation and other audio applications have traditionally
relied on the use of short-time Fourier transforms as a front-end
frequency domain representation step. The unavailability of a neural
network equivalent to forward and inverse transforms hinders the implementation
of end-to-end learning systems for these applications.

In this work, we present an auto-encoder neural network that can act as an equivalent
to short-time front-end transforms. We demonstrate the ability
of the network to learn optimal, real-valued basis functions directly
from the raw waveform of a signal and further show how it can be
used as an adaptive front-end for supervised source separation.

In terms of separation performance, these transforms significantly outperform 
their Fourier counterparts. Finally, we also propose a novelsource to distortion ratio 
based cost function for end-to-end sourceseparation.
