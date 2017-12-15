# Adaptive_front_ends
Code for the paper "Adaptive Front Ends for end-to-end source separation"

# Disclaimer
———————————————————
University of Illinois
Open Source License

Copyright © <Year>, <Organization Name>. All rights reserved.

Developed by:
Shrikant Venkataramani, Paris Smaragdis
University of Illinois at Urbana-Champaign, Adobe Research
This work was supported by NSF grant 1453104.
Paper link: https://arxiv.org/pdf/1705.02514.pdf

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal with the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimers.
Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimers in the documentation and/or other materials provided with the distribution.
Neither the names of Computational Audio Group, University of Illinois at Urbana-Champaign, nor the names of its contributors may be used to endorse or promote products derived from this Software without specific prior written permission.
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH THE SOFTWARE.
———————————————————

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
