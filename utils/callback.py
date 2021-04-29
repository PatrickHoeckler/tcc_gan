import numpy as np
import tensorflow as tf
import tensorflow_gan as tfgan
import matplotlib.pyplot as plt

from .fn import make_grid_image

"""
This callback was only used when training the classifiers used
in this project, it is called from the `tf.keras.Model.fit` method
"""
class SaveIfBestCallback(tf.keras.callbacks.Callback):
    """
        Saves the model which achieve the best validation accuracy at the
        specified path
    """
    def __init__(self, filename, save_after=0):
        """
            Parameters
            ----------
            filename : str
                Filename to save the best model
            save_after : int, optional
                Only starts saving after this number of epochs (Default is 0)
        """
        super().__init__()
        self.__max_accuracy = 0
        self.__filename = filename
        self.__save_after = save_after
      
    def on_epoch_end(self, epoch, logs=None):
        if (epoch < self.__save_after or logs['val_accuracy'] <= self.__max_accuracy):
            return
        self.__max_accuracy = logs['val_accuracy']
        self.model.save(self.__filename, overwrite=True, save_format='h5')


"""
The following callbacks were used when training the GANs, they are called directly
inside the training function written for the specific GAN. The methods that are
called are only `on_epoch_begin` and `on_epoch_end`. The parameters passed to
these methods are:
    generator : tf.keras.Model
    discriminator : tf.keras.Model
    epoch : int
"""

class TimerCallback(tf.keras.callbacks.Callback):
    """
        Keeps a list of intervals for each epoch. The interval starts
        counting when the method `on_epoch_begin` is called and it is
        appended to the list when `on_epoch_end` is called.
    """
    def __init__(self):
        self.__time = []
        
    def on_epoch_begin(self, **kwargs):
       self.__t0 = tf.timestamp()
        
    def on_epoch_end(self, **kwargs):
        self.__time.append(float(tf.timestamp() - self.__t0))

    def get_time(self):
        return self.__time


class SaveSamplesCallback(tf.keras.callbacks.Callback):
    """
        At the end of each epoch, saves a grid of the images produced by
        the generator when fed the given inputs.
    """
    def __init__(
        self,
        path_format,
        inputs,
        n_cols,
        grid_params=None,
        imshow_kwargs=None,
        savefig_kwargs=None,
        transform_samples=None
    ):
        """
            Parameters
            ----------
            path_format : str
                The format of the file to save the grid of images. The filename
                for any given epoch will be given by `path_format.format(epoch)`
            inputs : list, numpy.array, tensorflow.Tensor
                The inputs to feed to the generator to generate the images,
                must have the same shape as the input shape of the generator
            n_cols : int
                Number of columns of the grid of images produced
            grid_params : dict, optional
                Defines the parameters for the `make_grid_image` call, see this
                function documentation for details (Default is None).
                The dictionary keys are:
                    border : int
                    pad : int
                    pad_value : float
            imshow_kwargs : dict, optional
                Keyword arguments to send to the imshow call (Default is None).
            savefig_kwargs : dict, optional
                Keyword arguments to send to the savefig call (Default is None).
            transform_samples : function, optional
                Function to apply to the outputs of the generator (Default is None).
        """
        self.__path_format = path_format
        self.__inputs = inputs
        self.__n_cols = n_cols
        self.__grid_params = grid_params or {}
        self.__savefig_kwargs = savefig_kwargs or {}
        self.__imshow_kwargs = imshow_kwargs or {}
        self.__transform_samples = transform_samples

    def on_epoch_begin(self, **kwargs): return

    def on_epoch_end(self, epoch, generator, **kwargs):
        plt.ioff()
        samples = generator(self.__inputs, training=False)
        if self.__transform_samples:
            samples = self.__transform_samples(samples)
        grid = make_grid_image(samples, self.__n_cols, **self.__grid_params)

        fig = plt.figure()
        plt.imshow(grid, **self.__imshow_kwargs)
        plt.axis(False)
        fig.savefig(self.__path_format.format(epoch), **self.__savefig_kwargs)
        plt.close(fig)


class MetricsCallback(tf.keras.callbacks.Callback):
    """Calculate Frechet Distance and Classifier Score for a given classifier"""
    def __init__(
        self,
        generator,
        classifier,
        latent_dims,
        feature_layer,
        logits_layer,
        precalculated_features,
        n_samples=256*200,
        batch_size=256,
        save_after=None,
        save_to=None
    ):
        """
            Parameters
            ----------
            classifier : tf.keras.Model
            feature_layer : tf.keras.layers.Layer
                Layer of features of the classifier, the outputs will
                be used to calculate the Frechet Distance
            logits_layer : tf.keras.layers.Layer
                Logits layer of predictions of the classifier, the
                outputs will be used to calculate the Classifier Score
            precalculated_features : tf.Tensor or np.array
                Array of precalculated features of the dataset, will
                be used to calculate the Frechet Distance
            latent_dims : int
                Number of dimensions in the generator latent space
            n_samples : int
                Number of samples to generate in order to calculate
                the frechet distance and classifier score
            batch_size : int
                Batch size used to run the generator and classifier models
            save_after : int, optional
                If present it defines how many epochs must be run until
                saving the best generators found (default is None).
                If not None the argument `save_to` must include a filepath
                to save the model
            save_to : str, optional
                Only relevant when save_after is not None. Defines the
                path to save the best generator model. The best generator
                it the one with the lowest Frechet distance
        """
        self.__frechet_distance = []
        self.__classifier_score = []
        self.__precalculated_features = precalculated_features
        
        extractor = tf.keras.Model(
            inputs=classifier.input,
            outputs=[feature_layer.output, logits_layer.output]
        )
        samples = generator(generator.input)
        features, logits = extractor(samples)
        self.__reducer = tf.keras.Model(
            inputs=generator.input,
            outputs=[features, logits]
        )
        
        self.__latent_dims = latent_dims
        self.__n_samples = n_samples
        self.__batch_size = batch_size
        
        assert(save_after is None or type(save_to) == str)
        self.__save_after = save_after if save_after is not None else float('inf')
        self.__save_to = save_to
        self.__min_fcd = float('inf')

    def on_epoch_begin(self, **kwargs): return

    def on_epoch_end(self, epoch, generator, **kwargs):
        # seeds = tf.random.normal((self.__n_samples, self.__latent_dims))
        # features, logits = self.__reducer.predict(seeds, batch_size=self.__batch_size)
        inputs = self.get_random_inputs(self.__n_samples)
        features, logits = self.__reducer.predict(inputs, batch_size=self.__batch_size)
        
        fcd = tfgan.eval.frechet_classifier_distance_from_activations(features, self.__precalculated_features)
        cs = tfgan.eval.classifier_score_from_logits(logits)
        self.__frechet_distance.append(float(fcd))
        self.__classifier_score.append(float(cs))
        
        if (self.__save_after <= epoch and fcd < self.__min_fcd):
            self.__min_fcd = fcd
            generator.save(self.__save_to, overwrite=True, save_format='h5')

    def get_random_inputs(self, n_samples):
        return tf.random.normal((n_samples, self.__latent_dims))

    def get_metrics(self):
        return {
            'frechet_distance': self.__frechet_distance,
            'classifier_score': self.__classifier_score
        }

