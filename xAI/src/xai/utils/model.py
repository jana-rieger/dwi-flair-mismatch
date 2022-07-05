'''Model utilities'''

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras import Model
from termcolor import colored

from xAI.src.xai.network.GroupConv3D import GroupConv3D


def initiate_model(inputs, outputs):
    if inputs is not None and outputs is not None:
        return Model(inputs=inputs, outputs=outputs)
    return None


def load_models(model_path: str) -> tuple:
    '''
    Loads a Keras model.

    Args:
        model_path (str): Path to the model.

    Retuns:
        tuple(tf.keras.Model, tf.keras.Model): DWI and FLAIR models.
    '''

    # Load entire model
    model = load_model(model_path, custom_objects={'GroupConv3D': GroupConv3D,
                                                   # 'Functional': tf.keras.models.Model,
                                                   # 'HeNormal': tf.keras.initializers.he_normal(),
                                                   # 'L2': tf.keras.regularizers.l2
                                                   },
                       compile=False)

    # Separate imaging inputs and outputs
    layer_names = [layer.name for layer in model.layers]
    dwi_input = model.get_layer('dwi').input if 'dwi' in layer_names else None
    flair_input = model.get_layer('flair').input if 'flair' in layer_names else None
    dwi_output = model.get_layer('dwi_avg_pool').output if 'dwi_avg_pool' in layer_names else None
    flair_output = model.get_layer('flair_avg_pool').output if 'flair_avg_pool' in layer_names else None

    # Create DWI and FLAIR models
    dwi_model = initiate_model(inputs=dwi_input, outputs=dwi_output)
    flair_model = initiate_model(inputs=flair_input, outputs=flair_output)

    print('Model summary:')
    model.summary()
    if dwi_model is not None:
        dwi_model.summary()
    else:
        print(colored('No dwi model found.', 'red'))
    if flair_model is not None:
        flair_model.summary()
    else:
        print(colored('No flair model found.', 'red'))

    return dwi_model, flair_model

def load_entire_model(model_path: str) -> tf.keras.Model:
    '''
    Loads a Keras model.

    Args:
        model_path (str): Path to the model.

    Retuns:
        tf.keras.Model: Keras model.
    '''

    # Load entire model
    model = load_model(model_path, custom_objects={'GroupConv3D': GroupConv3D,
                                                   # 'Functional': tf.keras.models.Model,
                                                   # 'HeNormal': tf.keras.initializers.he_normal(),
                                                   # 'L2': tf.keras.regularizers.l2
                                                   },
                       compile=False)

    print('Model summary:')
    model.summary(line_length=125)

    return model

def load_bottleneck_model(model_path: str, layer_name: str) -> tf.keras.Model:
    '''
    Loads bottleneck model of a Keras U-Net model.

    Args:
        model_path (str): Path to the model.

    Retuns:
        tf.keras.Model: Keras model.
    '''

     # Load entire model
    model = load_model(model_path, compile=False)

    print('Model summary:')
    model.summary(line_length=125)

    layer_name = layer_name.lower().split()
    bottleneck_layer = None
    for layer in reversed(model.layers):
        if all([word in layer.name.lower() for word in layer_name]):
            bottleneck_layer = layer
            break

    if bottleneck_layer:
        bottleneck_model = Model(inputs=model.inputs, outputs=bottleneck_layer.output)

        if K.ndim(bottleneck_model.output) > 2:
            model_output = tf.keras.layers.Flatten()(bottleneck_model.output)
            bottleneck_model = Model(inputs=model.inputs, outputs=model_output)

        print('Bottleneck model summary:')
        bottleneck_model.summary(line_length=125)
        return bottleneck_model
    else:
        raise ValueError("Could not find the layer.")

def last_layer_to_linear(model: tf.keras.Model, index: int = -1, clone: bool = False) -> tf.keras.Model:
    '''
    Replaces a softmax function of the last layer to a linear function.

    Args:
        model (tf.keras.Model): Keras model.
        index (int): Index of the layer, in case rather than last layer.
        clone (bool): Wether to clone the original model.

    Returns:
        tf.keras.Model: A model without softmax.
    '''

    def to_linear(model: tf.keras.Model, index: int):
        '''
        Replaces a softmax function of the last layer to a linear function.

        Args:
            model (tf.keras.Model): Keras model.
            index (int): Index of the layer.
        '''

        model.layers[index].activation = tf.keras.activations.linear

    if clone:
        new_model = tf.keras.models.clone_model(model)
        new_model.set_weights(model.get_weights())
    else:
        new_model = model

    to_linear(new_model, index)

    return new_model
