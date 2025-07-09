"""
Created on 09.08.2018 

@author: goh
"""
import os.path

from tensorflow.keras.layers import Dense, Dropout, Conv2D, Conv1D, MaxPooling1D, MaxPooling2D, Flatten, \
    BatchNormalization, Activation, GaussianNoise, GlobalMaxPooling2D, GlobalAveragePooling2D, Input, Add, Layer, AveragePooling2D, Lambda
from tensorflow.keras.models import Sequential, Model, model_from_json
from tensorflow.keras.backend import max, mean
from tensorflow.keras.regularizers import l2, l1
import tensorflow.keras.backend as K
import tensorflow as tf
from ..logging_alf import getLogger
from ..core.helper.helper import get_method_from_string

LOGGER = getLogger()


def createDNNModel(shape_in, num_output_dim, **modelParams):
    """Example function for generating dnn model using parameters.

        Valid parameters are:
            layer_dims: number of layers with units without output [128, 64, 32]
            dropout_input: dropout between input and first layer
            dropout: dropout between hidden layers
            dropout_output: dropout before output layer
            gaussian_noise_input: gaussian noise std dev for input
            l2: l2 reg in all hidden layers
            l1: l1 reg in all hidden layers -> only l2 or l1 can be set
            kernel_initializer: init of layers
            activation: activation function to use in hidden layers
            activation_output: activation function for output layer

        Args:
            shape_in (np.shape): shape of features (alf interface)
            num_output_dim (int): number of output units (alf interface)
            modelParams (**kwargs): dict with additional parameters (alf interface)
        Returns:
            keras model: created model (alf interface)
        """

    layer_dims = modelParams["layer_dims"] if "layer_dims" in modelParams else []
    dropout_input = modelParams["dropout_input"] if "dropout_input" in modelParams else 0
    dropout = modelParams["dropout"] if "dropout" in modelParams else 0
    dropout_output = modelParams["dropout_output"] if "dropout_output" in modelParams else 0
    gauss_in = modelParams["gaussian_noise_input"] if "gaussian_noise_input" in modelParams else 0
    l2Reg = modelParams["l2"] if "l2" in modelParams else 0
    l1Reg = modelParams["l1"] if "l1" in modelParams else 0
    kernel_initializer = modelParams["kernel_initializer"] if "kernel_initializer" in modelParams else tf.keras.initializers.HeUniform()
    activation = modelParams["activation"] if "activation" in modelParams else "relu"
    activation_output = modelParams["activation_output"] if "activation_output" in modelParams else "softmax"

    if len(layer_dims) < 1:
        LOGGER.error("at least one hidden layer must be set")
        raise RuntimeError

    if l1Reg > 0 and l2Reg > 0:
        LOGGER.error("Only l1 or l2 reg can be used (>0)")
        raise RuntimeError

    reg = None
    if l2Reg > 0:
        reg = l2(l2Reg)
    elif l1Reg > 0:
        reg = l1(l1Reg)

    num_layer = len(layer_dims)
    model = Sequential()

    if gauss_in > 0:
        model.add(GaussianNoise(gauss_in, input_shape=(shape_in[1],)))

    if dropout_input > 0:
        model.add(Dropout(dropout_input, input_shape=(shape_in[1],)))

    model.add(Dense(input_dim=int(shape_in[1]), units=layer_dims[0], kernel_regularizer=reg,
                    activation=activation,
                    kernel_initializer=kernel_initializer))

    for i in range(1, num_layer):
        if dropout > 0:
            model.add(Dropout(dropout))

        model.add(Dense(units=layer_dims[i], kernel_regularizer=reg,
                        activation=activation,
                        kernel_initializer=kernel_initializer))

    # add output layer
    if dropout_output > 0:
        model.add(Dropout(dropout_output))

    model.add(Dense(units=num_output_dim, activation=activation_output))
    return model


def createModelCNN2D(shape_in, num_output_dim, **modelParams):
    """Example function for generating cnn model using parameters.

        Valid parameters are:
            layer_dims: number of layers with units without output,
                        for conv [num filters, [kernelsize*kernelsize]]
                        for dense [num units]
                         [[8, [3, 3]], [8, [3, 3]], 16]
            gaussian_noise_input: gaussian noise std dev for input
            dropout: dropout between hidden layers
            dropout_output: dropout before output layer
            l2: l2 reg in all hidden layers
            l1: l1 reg in all hidden layers -> only l2 or l1 can be set
            kernel_initializer: init of layers
            activation: activation function to use in hidden layers
            activation_output: activation function for output layer
            addPooling: add max addPooling (2,2) after conv
            batchNorm: add batch norm between conv
            padding: padding of conv
            final_global_pooling: if set global pooling will be done after last layer
                    -> "max" for global max, "avg" for global average pooling
        Args:
            shape_in (np.shape): shape of features (alf interface)
            num_output_dim (int): number of output units (alf interface)
            modelParams (**kwargs): dict with additional parameters (alf interface)
        Returns:
            keras model: created model (alf interface)
        """
    pool_only_in_time = modelParams.get("pool_only_in_time", False)

    if pool_only_in_time is True:
        maxPoolingDim = (2, 1)
    else:
        maxPoolingDim = (2, 2)

    return __createGeneralConvModel(shape_in, num_output_dim, Conv2D, MaxPooling2D, maxPoolingDim, **modelParams)


def createModelCNN1D(shape_in, num_output_dim, **modelParams):
    return __createGeneralConvModel(shape_in, num_output_dim, Conv1D, MaxPooling1D, 2, **modelParams)


def __createGeneralConvModel(shape_in, num_output_dim, convLayer, maxPooling, maxPoolingDim, **modelParams):
    layer_dims = modelParams["layer_dims"] if "layer_dims" in modelParams else []
    # No droput_in --> https://www.quora.com/Why-would-I-need-to-apply-a-dropout-layer-before-a-convolutional-layer
    gauss_in = modelParams["gaussian_noise_input"] if "gaussian_noise_input" in modelParams else 0
    dropout = modelParams["dropout"] if "dropout" in modelParams else 0
    dropout_output = modelParams["dropout_output"] if "dropout_output" in modelParams else 0
    l2Reg = modelParams["l2"] if "l2" in modelParams else 0
    l1Reg = modelParams["l1"] if "l1" in modelParams else 0
    kernel_initializer = modelParams["kernel_initializer"] if "kernel_initializer" in modelParams else tf.keras.initializers.HeUniform()
    activation = modelParams["activation"] if "activation" in modelParams else "relu"
    addMaxPooling = modelParams["addPooling"] if "addPooling" in modelParams else False
    batchNorm = modelParams["batchNorm"] if "batchNorm" in modelParams else False
    activation_output = modelParams["activation_output"] if "activation" in modelParams else "softmax"
    padding = modelParams["padding"] if "padding" in modelParams else "same"
    final_global_pooling = modelParams["final_global_pooling"] if "final_global_pooling" in modelParams else None
    pool_only_in_time = modelParams.get("pool_only_in_time", False)

    if len(layer_dims) < 1:
        LOGGER.error("at least one hidden layer must be set")
        raise RuntimeError

    if len(layer_dims[0]) < 2:
        LOGGER.error("first layer dim must be [filter, [kernel]] for conv!")
        raise RuntimeError

    if l1Reg > 0 and l2Reg > 0:
        LOGGER.error("Only l1 or l2 reg can be used (>0)")
        raise RuntimeError

    if final_global_pooling is False:
        final_global_pooling = None
    # for compatibility with existing files
    elif final_global_pooling is True:
        LOGGER.debug("Using global max pooling")
        final_global_pooling = "max"

    # mean or average should both work
    if final_global_pooling == "mean":
        final_global_pooling = "avg"

    if final_global_pooling is not None and final_global_pooling != "max" and final_global_pooling != "avg":
        LOGGER.error("Unknown final pooling type")
        raise RuntimeError

    reg = None
    if l2Reg > 0:
        reg = l2(l2Reg)
    elif l1Reg > 0:
        reg = l1(l1Reg)

    model = Sequential()
    input_shape = shape_in[1:]

    if input_shape is None:
        LOGGER.error("Dimension of Input_Shape is incorrect!")
        raise ValueError

    for i, layer in enumerate(layer_dims):
        # conv
        if type(layer) == list:
            # TODO required?
            if len(shape_in) == 4:
                kernelSize = (layer[1][0], layer[1][1])
            else:
                kernelSize = layer[1][0]

            # add input shape for first layer
            if i == 0:
                if gauss_in > 0:
                    model.add(GaussianNoise(gauss_in, input_shape=input_shape))

                model.add(
                    convLayer(layer[0], kernel_size=kernelSize, input_shape=input_shape, data_format='channels_last',
                              padding=padding, kernel_regularizer=reg, kernel_initializer=kernel_initializer))
            else:
                model.add(convLayer(layer[0], kernel_size=kernelSize, padding=padding, data_format='channels_last',
                                    kernel_regularizer=reg, kernel_initializer=kernel_initializer))

            if batchNorm is True:
                model.add(BatchNormalization())

            model.add(Activation(activation))

            if i + 1 < len(layer_dims):
                isLastConvLayer = not isinstance(layer_dims[i + 1], list)
            else:  # last layer in general?
                isLastConvLayer = len(layer_dims) == (i + 1)

            if final_global_pooling == "max" and isLastConvLayer is True:
                if pool_only_in_time is False:
                    model.add(GlobalMaxPooling2D(data_format='channels_last', name="final_global_pooling"))
                else:
                    model.add(Lambda(lambda x: max(x, axis=1)))
                    model.add(Flatten())
            elif final_global_pooling == "avg" and isLastConvLayer is True:
                if pool_only_in_time is False:
                    model.add(GlobalAveragePooling2D(data_format='channels_last', name="final_global_pooling"))
                else:
                    model.add(Lambda(lambda x: mean(x, axis=1)))
                    model.add(Flatten())
            # flatten after final conv if needed and not global pooled
            elif final_global_pooling is None and isLastConvLayer is True:
                if addMaxPooling is True:
                    model.add(maxPooling(maxPoolingDim, data_format='channels_last'))
                model.add(Flatten())
            elif addMaxPooling is True:
                model.add(maxPooling(maxPoolingDim, data_format='channels_last'))

            # no dropout after final conv
            if dropout > 0 and isLastConvLayer is False:
                model.add(Dropout(dropout))
        # dense after conv if set
        else:
            if dropout > 0:
                model.add(Dropout(dropout))
            model.add(Dense(layer, kernel_regularizer=reg, activation=activation, kernel_initializer=kernel_initializer))

    if dropout_output > 0:
        model.add(Dropout(dropout_output))

    # classification layer
    model.add(Dense(units=num_output_dim, activation=activation_output))

    return model


def load_pretrained_model(shape_in, num_output_dim, **params):
    # freeze options are None or number of layers (starting at end) that should be trained
    model_path = params.get("path", "")
    freeze_layers = params.get("freeze_layers", None)
    activation_output = params.get("activation_output", "softmax")

    if not os.path.exists(model_path):
        raise RuntimeError("Model not found in path")

    model = tf.keras.models.load_model(model_path)

    # remove classification layer
    x = model.layers[-2].output

    # add new classification layer
    x = tf.keras.layers.Dense(num_output_dim, activation=activation_output)(x)
    classification_model = tf.keras.Model(inputs=model.inputs, outputs=x)

    # reset it (state stays same otherwise)
    model.trainable = True

    if freeze_layers != None:
        # train also last conv
        model.trainable = False

        # don't count last Dense layer
        freeze_layers += 1

        for layer in model.layers[::-1]:
            layer.trainable = True
            freeze_layers -= 1
            if freeze_layers == 0:
                break

    return classification_model


def load_model_from_json(shape_in, num_output_dim, **modelParams):
    file = modelParams.get("file", None) # path json file
    custom_objects = modelParams.get("custom_objects", None) # dict with all custom object in json "custom_objects": {"Addons>GroupNormalization": "tensorflow_addons.layers.GroupNormalization"}
    keep_output = modelParams.get("keep_output", False)  # if true original output will be kept, otherwise it will be replace with current model data (default)
    activation_output = modelParams.get("activation_output", "softmax")

    if file is None:
        raise ValueError("file parameter pointing to model file must be set")
    if not os.path.exists(file):
        raise RuntimeError("json model not found")

    json_file = open(file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    if custom_objects is not None:
        for key, val in custom_objects.items():
            custom_objects[key] = get_method_from_string(val)

    model = model_from_json(loaded_model_json, custom_objects)

    # input must fit but we can adjust output
    if keep_output is True:
        return model
    else:
        x = model.get_layer(index=-2).output
        out = Dense(units=num_output_dim, activation=activation_output, name="adjusted_output")(x)
        new_model = Model(inputs=model.inputs, outputs=out)
        return new_model


def createResNetKoutiniRN1(shape_in, num_output_dim, **modelParams):
    """
    create model RN1 according to paper https://arxiv.org/abs/1907.01803
    """
    modelParams["first_conv"] = [128, 5]
    modelParams["res_blocks"] = [[128, 3, 1], "p", [128, 3, 3], "p", [128, 3, 3], [256, 3, 3], "p", [512, 3, 1]]
    modelParams["final_global_pooling"] = "avg"
    return createResNet(shape_in, num_output_dim, **modelParams)


def createResNet(shape_in, num_output_dim, **modelParams):
    """Example function for generating resnet model using parameters.

            Valid parameters are:
                first_conv: first conv layer with filters & kernels (ex. [128, 5])
                res_blocks: sequential res net blocks with filters, kernel1, kernel2 and pooling#
                    (ex. [[128, 3, 1], "p", [128, 3, 3], "p", ... "p", [128, 3, 1])
                version: which versions of resnet blocks:
                    "v1" = ResNetV1Block
                    "ic" = ResNetICBlock
                    "shake" = ResNetShakeShakeBlock
                    "shake_small" = ResNetShakeShakeBlockSmall
                    "wide" = ResNetWideBlock
                activation: activation function to use in hidden layers
                activation_output: activation function for output layer
                pooling_version: max for max pooling in res blocks, avg for average pooling
                final_global_pooling: if set global pooling will be done after last layer
                        -> "max" for global max, "avg" for global average pooling
                l2: l2 kernel reg value in all conv layers
                ic_dropout: only for "ic" version: dropout ratio in IC Layers
                residual_dropout: only for "ic" version: add dropout also to residual connection
            Args:
                shape_in (np.shape): shape of features (alf interface)
                num_output_dim (int): number of output units (alf interface)
                modelParams (**kwargs): dict with additional parameters (alf interface)
            Returns:
                keras model: created model (alf interface)
            """
    final_global_pooling = modelParams["final_global_pooling"] if "final_global_pooling" in modelParams else None
    if final_global_pooling != "avg" and final_global_pooling != "max":
        raise ValueError("final_global_pooling must be set to avg or max")

    first_conv = modelParams["first_conv"] if "first_conv" in modelParams else None
    if first_conv is None or type(first_conv) != list or len(first_conv) != 2:
        raise ValueError("first_conv must be set with [filters, kernel]")

    res_blocks = modelParams["res_blocks"] if "res_blocks" in modelParams else None
    if res_blocks is None or type(res_blocks) != list or len(res_blocks) < 1:
        raise ValueError(
            "res_blocks must be set with [[filters, kernel1, kernel2], \"p\", [filters, kernel1, kernel2]] with p for max pooling")
            
    l2_val = modelParams.get("l2", 0)

    version = modelParams["version"] if "version" in modelParams else "v1"
    activation = modelParams["activation"] if "activation" in modelParams else "relu"
    if activation == "hswish":
        activation = hswish
    output_activation = modelParams["output_activation"] if "output_activation" in modelParams else "softmax"
    pooling_version = modelParams.get("pooling_version", "avg")
    
    ic_dropout = modelParams.get("ic_dropout", 0.1)
    if ic_dropout > 1.0:
        raise ValueError("ic_dropout cannot be bigger than one")
    residual_dropout = modelParams.get("residual_dropout", False)

    inputs = Input(shape=shape_in[1:])
    x = Conv2D(first_conv[0], first_conv[1], strides=2, activation=activation, padding="same", use_bias=False,
               kernel_initializer=tf.keras.initializers.HeUniform(), kernel_regularizer=tf.keras.regularizers.l2(l2_val))(inputs)
    x = BatchNormalization()(x)

    for rb in res_blocks:
        if type(rb) == list:
            if len(rb) != 3:
                raise ValueError("res block must contain [filters, kernel1, kernel2]")
            if version == "v1":
                x = __ResNetV1Block(x, rb[0], rb[1], rb[2], activation, l2_val)
            elif version == "ic":
                x = __ResNetICBlock(x, rb[0], rb[1], rb[2], activation, l2_val, ic_dropout, residual_dropout)
            elif version == "shake":
                x = __ResNetShakeShakeBlock(x, rb[0], rb[1], rb[2], activation, l2_val)
            elif version == "shake_small":
                x = __ResNetShakeShakeBlockSmall(x, rb[0], rb[1], rb[2], activation, l2_val)
            elif version == "wide":
                x = __ResNetWideBlock(x, rb[0], rb[1], rb[2], activation, l2_val)
            else:
                raise ValueError("Unknown resnet version. Must be v1, ic, shake, shake_small or wide")
        elif rb == "p" and pooling_version == "max":
            x = MaxPooling2D()(x)
        elif rb == "p" and pooling_version == "avg":
            x = AveragePooling2D()(x)
        elif rb == "p" and pooling_version is not None:
            raise ValueError("Unknown pooling mode, use max or avg")
        else:
            raise ValueError("Unknown element in list", rb)

    if final_global_pooling == "avg":
        x = GlobalAveragePooling2D()(x)
    else:
        x = GlobalMaxPooling2D()(x)

    # classification layer
    outputs = Dense(num_output_dim, activation=output_activation)(x)
    model = Model(inputs=inputs, outputs=outputs, name="res_net")
    return model


def __ResNetV1Block(inputs, filters, ks1, ks2, activation, l2_val):
    x = Conv2D(filters, ks1, padding="same", use_bias=False, kernel_initializer=tf.keras.initializers.HeUniform(), kernel_regularizer=tf.keras.regularizers.l2(l2_val))(inputs)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)

    x = Conv2D(filters, ks2, padding="same", use_bias=False, kernel_initializer=tf.keras.initializers.HeUniform(), kernel_regularizer=tf.keras.regularizers.l2(l2_val))(x)
    x = BatchNormalization()(x)

    # identity block
    if inputs.shape[-1] != filters:
        inputs = Conv2D(filters, 1, padding="same", use_bias=False, kernel_initializer=tf.keras.initializers.HeUniform())(inputs)

    # shortcut
    x = Add()([x, inputs])
    
    x = Activation(activation)(x)

    return x


def __ResNetWideBlock(inputs, filters, ks1, ks2, activation, l2_val):
    x = BatchNormalization()(inputs)
    x = Activation(activation)(x)
    x = Conv2D(filters, ks1, padding="same", use_bias=False, kernel_initializer=tf.keras.initializers.HeUniform(), kernel_regularizer=tf.keras.regularizers.l2(l2_val))(x)

    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv2D(filters, ks2, padding="same", use_bias=False, kernel_initializer=tf.keras.initializers.HeUniform(), kernel_regularizer=tf.keras.regularizers.l2(l2_val))(x)

    # identity block
    if inputs.shape[-1] != filters:
        inputs = Conv2D(filters, 1, padding="same", use_bias=False, kernel_initializer=tf.keras.initializers.HeUniform())(inputs)

    # shortcut
    x = Add()([x, inputs])

    return x


def __ResNetICBlock(inputs, filters, ks1, ks2, activation, l2_val, ic_dropout, residual_dropout):
    """
    IC layer (batch norm + dropout) taken from https://arxiv.org/pdf/1905.05928
    with Relu - IC - Conv scheme
    """
    x = Activation(activation)(inputs)
    x = __ICLayer(x, ic_dropout)
    x = Conv2D(filters, ks1, padding="same", use_bias=False, kernel_initializer=tf.keras.initializers.HeUniform(), kernel_regularizer=tf.keras.regularizers.l2(l2_val))(x)

    x = Activation(activation)(x)
    x = __ICLayer(x, ic_dropout)
    x = Conv2D(filters, ks2, padding="same", use_bias=False, kernel_initializer=tf.keras.initializers.HeUniform(), kernel_regularizer=tf.keras.regularizers.l2(l2_val))(x)

    # identity block
    if inputs.shape[-1] != filters:
        if residual_dropout is True:
            inputs = Dropout(ic_dropout)(inputs)
        inputs = Conv2D(filters, 1, padding="same", use_bias=False, kernel_initializer=tf.keras.initializers.HeUniform())(inputs)

    # shortcut
    x = Add()([x, inputs])

    return x


def __ICLayer(inputs, ic_dropout):
    x = BatchNormalization()(inputs)
    x = Dropout(ic_dropout)(x)
    return x


def __ResNetShakeShakeBlock(inputs, filters, ks1, ks2, activation, l2_val):
    # shake shake block as in Koutini2020_music_tagging
    x1 = Conv2D(filters, ks1, padding="same", use_bias=False, kernel_initializer=tf.keras.initializers.HeUniform(), kernel_regularizer=tf.keras.regularizers.l2(l2_val))(inputs)
    x1 = BatchNormalization()(x1)
    x1 = Activation(activation)(x1)

    x1 = Conv2D(filters, ks2, padding="same", use_bias=False, kernel_initializer=tf.keras.initializers.HeUniform(), kernel_regularizer=tf.keras.regularizers.l2(l2_val))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation(activation)(x1)

    x2 = Conv2D(filters, ks1, padding="same", use_bias=False, kernel_initializer=tf.keras.initializers.HeUniform(), kernel_regularizer=tf.keras.regularizers.l2(l2_val))(inputs)
    x2 = BatchNormalization()(x2)
    x2 = Activation(activation)(x2)

    x2 = Conv2D(filters, ks2, padding="same", use_bias=False, kernel_initializer=tf.keras.initializers.HeUniform(), kernel_regularizer=tf.keras.regularizers.l2(l2_val))(x2)
    x2 = BatchNormalization()(x2)
    x2 = Activation(activation)(x2)

    # identity block
    if inputs.shape[-1] != filters:
        inputs = Conv2D(filters, 1, padding="same", use_bias=False, kernel_initializer=tf.keras.initializers.HeUniform())(inputs)

    # shake and shortcut
    x = Add()([inputs, ShakeShake()([x1, x2])])

    return x


def __ResNetShakeShakeBlockSmall(inputs, filters, ks1, ks2, activation, l2_val):
    # smaller res net block with shake shake reg but only one Conv in each branch
    x1 = Conv2D(filters, ks1, padding="same", use_bias=False, kernel_initializer=tf.keras.initializers.HeUniform(), kernel_regularizer=tf.keras.regularizers.l2(l2_val))(inputs)
    x1 = BatchNormalization()(x1)
    x1 = Activation(activation)(x1)

    x2 = Conv2D(filters, ks2, padding="same", use_bias=False, kernel_initializer=tf.keras.initializers.HeUniform(), kernel_regularizer=tf.keras.regularizers.l2(l2_val))(inputs)
    x2 = BatchNormalization()(x2)
    x2 = Activation(activation)(x2)

    # identity block
    if inputs.shape[-1] != filters:
        inputs = Conv2D(filters, 1, padding="same", use_bias=False, kernel_initializer=tf.keras.initializers.HeUniform())(inputs)

    # shake and shortcut
    x = Add()([inputs, ShakeShake()([x1, x2])])

    return x


def hswish(x):
    """ Faster swish approximation (hard swish) using Relu6 from MobileNetV3: https://arxiv.org/pdf/1905.02244.pdf """
    return x * tf.nn.relu6(x+3) * 0.16666667

class ShakeShake(Layer):
    """ Shake-Shake Layer from https://github.com/jonnedtc/Shake-Shake-Keras """

    def __init__(self, **kwargs):
        super(ShakeShake, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ShakeShake, self).build(input_shape)

    def call(self, x, **kwargs):
        # unpack x1 and x2
        assert isinstance(x, list)
        x1, x2 = x
        # create alpha and beta
        batch_size = K.shape(x1)[0]
        alpha = K.random_uniform((batch_size, 1, 1, 1))
        beta = K.random_uniform((batch_size, 1, 1, 1))

        # shake-shake during training phase
        def x_shake():
            return beta * x1 + (1 - beta) * x2 + K.stop_gradient((alpha - beta) * x1 + (beta - alpha) * x2)

        # even-even during testing phase
        def x_even():
            return 0.5 * x1 + 0.5 * x2

        return K.in_train_phase(x_shake, x_even)


def compute_output_shape(input_shape):
    assert isinstance(input_shape, list)
    return input_shape[0]
