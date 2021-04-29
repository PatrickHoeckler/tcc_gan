# An Analysis of Techniques for Building Generative Adversarial Networks
Everything related with my graduation project "An Analysis of Techniques for Building Generative Adversarial Networks"

The document can be found at: [Will be here soon](.)

## Repository Structure
### Documentation
  + **Overleaf** - all files used in the overleaf project to produce the final document.
  + **drawio** - files for the diagrams used in the document, can be opened and edited at [draw.io](https://app.diagrams.net/).
  + **images** - images referenced in the README files in this repository

### Code
All code except for the ones contained in the `utils` directory is written in Jupyter notebooks.
  + **utils** - python module for custom functions frequently used in other parts of the code.
  + **Classifiers** - implementation of the MNIST, Fashion MNIST, and CIFAR-10 classifiers. These are used to calculate the Classifier Score (CS) and Fréchet Classifier Distance (FCD) as described in the document in section `4.4 EVALUATING GANS`.
  + **GAN** - Code related to the simple GAN implementation described in sections `4.2 THE GAN ARCHITECTURE` and `5.1 SIMPLE GAN` of the document.
  + **DCGAN** - Code related to the Deep Convolutional GAN (DCGAN) implementation described in subsection `4.3.1 DCGAN` and section `5.2 DCGAN` of the document.
  + **CGAN** - Code related to the Conditional GAN (CGAN) implementation described in subsection `4.3.2 Conditional GAN` and section `5.3 CGAN` of the document.
  + **WGAN** - Code related to the Wasserstein GAN (WGAN) implementation described in subsection `4.3.3 Wasserstein GAN` and section `5.4 WGAN` of the document.
  + **WGAN-GP** - Code related to the WGAN with Gradient Penalty (WGAN-GP) implementation described in subsection `4.3.4 WGAN with Gradient Penalty` and section `5.5 WGAN-GP` of the document.
  + **Other** - Code for creating the different visualizations shown in the document.