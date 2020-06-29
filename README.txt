This project contains the code used in the simulations of Section 5 of the paper "A Provably Convergent and Practical Algorithm for Min-max Optimization with Applications to GANs" https://arxiv.org/abs/2006.12376

--The file "MNIST_Code.ipynb" contains the code used for the simulations in Figure 2, and in Appendix E.1, E.3, E.4, and E.5 of our paper.  (This file contains the code for our algorithm and for GDA, on the MNIST dataset.)

--The file "CIFAR_Code.ipynb" contains the code used for the simulations in Figure 1 and in Appendix E.2 of our paper.  (This file contains the code for our algorithm and for GDA, on the CIFAR-10 dataset.)

--The file "Gaussian_Mixture_code.ipynb" contains the code for the simulations in Figure 3 and Appendix E.6 of our paper.  (This file contains the code for our algorithm, GDA, and Unrolled GANs, on the four Gaussian mixture dataset.)

--The file "MNIST_decreasing_temperature_Code.ipynb" contains the code that was used to generate the results in Appendix E.5 of our paper. (This file contains the code for the version of our algorithm with randomized accept/reject step and decreasing temperature schedule.)



Our code for the MNIST simulations is based on the code of Renu Khandelwal https://medium.com/datadriveninvestor/generative-adversarial-network-gan-using-keras-
ce1c05cfdfd3 and Rowel Atienza https://towardsdatascience.com/gan-by-example-using-keras-on-tensorflow-backend-1a6d515a60d0.

Our code for the CIFAR simulations is based on the code of Jason Brownlee https://machinelearningmastery.com/how-to-develop-a-generative-adversarial- network-for-a-cifar-10-small-object-photographs-from-scratch/

Our code for the Gaussian Simulations is based on code provided by Luke Metz, Ben Poole, David Pfau, and Jascha Sohl-Dickstein for their paper Unrolled generative adversarial networks. https://arxiv.org/abs/1611.02163
