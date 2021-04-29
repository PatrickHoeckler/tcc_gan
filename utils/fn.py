import numpy as np
import tensorflow as tf
import json


def make_grid_image(images, n_cols, border=0, pad=0, pad_value=0.0):
    """
        Given a set of images from a numpy array or tensorflow Tensor in the
        shape (n_images, height, width, channels), returns a numpy array
        representing a single image with shape (height, width, channels)
        containing all the images given in a grid with the specified number
        of columns and optionally with a border and/or padding.

        Parameters
        ----------
        images : numpy.array, tensorflow.Tensor
            Set of images to put in the grid, must be of shape 
            (n_images, height, width, channels)
        n_cols : int
            Number of columns in the grid. If the columns don't evenly divide
            the number of images then the last row of the grid will only be
            partially filled
        border : int, optional
            How many pixels will the border have (Default is 0)
        pad : int, optional
            How many pixels will the padding between images have (Default is 0)
        pad_value : float
            The value of the pixels behind the images, that is, the border,
            padding, and pixels of an unfilled row
        
        Returns
        -------
        Numpy array of shape (height, width, channels) containing all images
        in a grid
    """
    n_images, height, width, channels = images.shape
    n_rows = 1 + (n_images - 1) // n_cols
    out = pad_value * np.ones((
        n_rows*height + (n_rows - 1)*pad + 2*border,
        n_cols*width  + (n_cols - 1)*pad + 2*border,
        channels
    ))
    for i, img in enumerate(images):
        row = i // n_cols
        col = i % n_cols
        drow = (row)*pad + border
        dcol = (col)*pad + border
        out[
            drow + row*height : drow + (row+1)*height,
            dcol + col*width  : dcol + (col+1)*width
        ] = img
    return out


def calculate_features(classifier, feature_layer, x, batch_size=256):
    """
        Given a classifier model, a feature layer belonging to this model,
        and a set of inputs, calculates all the outputs of the feature layer
        for all inputs.

        Parameters
        ----------
        classifier : tf.keras.Model
        feature_layer : tf.keras.layers.Layer
        x : numpy.array, tensorflow.Tensor
            Set of inputs to calculate the features from
        batch_size : int, optional
            Size of batches to calculate the features (Default is 256)
        
        Returns
        -------
        Numpy array of features
    """
    model = tf.keras.Model(inputs=classifier.inputs, outputs=feature_layer.output)
    features = model.predict(x, batch_size)
    return features


def update_json_log(filename, obj):
    """
        Opens a JSON file and updates the keys of the main object with
        the keys in the given object. If the file does not exist, then
        creates it and saves the object in JSON format.

        Parameters
        ----------
        filename : str
        obj : dict
    """
    with open(filename, 'a+') as file:
        file.seek(0)
        try:
            data = json.load(file)
        except:
            data = {}
        data = {**data, **obj}
        file.truncate(0)
        json.dump(data, file, indent='\t')