These files will build and train the classifiers that will be used to calculate the *Classifier Score* (CS) and *Fréchet Classifier Distance* (FCD) for all GAN models. So before training the GANs, make sure to train the classifiers first in order to calculate the metrics.

Each file will create a `.h5` file corresponding to the model trained, for example, `MNIST.ipynb` will create a `mnist.h5` file containing the MNIST classifier.