import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout, Conv2D, MaxPooling2D, AveragePooling2D, \
    SeparableConv2D, DepthwiseConv2D, BatchNormalization, SpatialDropout2D, Input, Flatten, GaussianNoise, ConvLSTM2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K
import tensorflow.compat.v1 as tf


from tensorflow.keras import Input, layers




tf.disable_v2_behavior()


# sess = tf.Session()
def shape(tensor):
    s = tensor.get_shape()
    return tuple([s[i].value for i in range(0, len(s))])



def DeepConvNet(nb_classes, Chans=64, Samples=128,
                dropoutRate=0.5, kernLength=64, F1=8,
                D=2, F2=16, norm_rate=0.25, EnK=True, dropoutType='Dropout'):
    # start the model
    input_main = Input((1, Chans, Samples))

    block1 = Conv2D(25, (1, 5),
                    input_shape=(1, Chans, Samples), padding='same',
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)), data_format='channels_first')(input_main)

##    # Using EnK Layer
##    block1 = EnKLayer(filter=25, EnK=EnK, InputData_shape=(1, Chans, Samples),
##                      output_dimension=(1, 1, 5), conv2DOutput=block1)(input_main)
    block1 = Conv2D(25, (Chans, 1),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)), data_format='channels_first')(block1)
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), data_format='channels_first')(block1)
    block1 = Dropout(dropoutRate)(block1)

    block2 = Conv2D(50, (1, 5),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)), data_format='channels_first')(block1)
    block2 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)
    block2 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), data_format='channels_first')(block2)
    block2 = Dropout(dropoutRate)(block2)

    block3 = Conv2D(100, (1, 5),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)), data_format='channels_first')(block2)
    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)
    block3 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), data_format='channels_first')(block3)
    block3 = Dropout(dropoutRate)(block3)

    block4 = Conv2D(200, (1, 5),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)), data_format='channels_first')(block3)
    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)
    block4 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), data_format='channels_first')(block4)
    block4 = Dropout(dropoutRate)(block4)

    flatten = Flatten(data_format='channels_first')(block4)

    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)


