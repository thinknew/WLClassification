# Ref: https://github.com/AdalbertoCq/Deep-Learning-Specialization-Coursera/blob/master/Convolutional%20Neural%20Networks/week1/convolution_model.py
# Custom Conv Layer: https://stackoverflow.com/questions/54093950/how-to-experiment-with-custom-2d-convolution-kernels-in-keras
# Alternate of For loop ? https://www.tensorflow.org/api_docs/python/tf/einsum


from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
import tensorflow.compat.v1 as tf
from im2col import *
from tensorflow.keras.layers import Conv2D


def shape(tensor):
    s = tensor.get_shape()
    return tuple([s[i].value for i in range(0, len(s))])

# Using Tenssorflow Conv2d Function and adding EnKernel methodology
class MyConv2D_TF(Layer):  # With Leanrable Parameters

    def __init__(self, filter, EnK,InputData_shape,output_dimension,dynamic=True):

        self.filter = filter
        self.output_dimension = output_dimension
        self.EnK =EnK
        self.InputData_shape=InputData_shape
        # self.encoding_vector= encoding_vector
        super(MyConv2D_TF, self).__init__()

    def build(self, input_shape):
        # Kernel dimension [filter_height, filter_width, in_channels, out_channels]
        self.kernel = self.add_weight(name='kernel', shape=(self.output_dimension[1],
                                                            self.output_dimension[2],self.output_dimension[0],self.filter),
                                      initializer='ones',trainable=False)

        if self.EnK:
            self.scaleFactor = self.add_weight(name='scaleFactor',shape=(1,), initializer='zeros', trainable=True)
        else:
            self.scaleFactor = 0

        super(MyConv2D_TF, self).build(input_shape)

    def call(self, input):

        Part1 = Conv2D(self.filter, (self.output_dimension[1], self.output_dimension[2]), padding='same',
                        input_shape=(self.InputData_shape[0], self.InputData_shape[1],self.InputData_shape[2]),
                        use_bias=False, data_format='channels_first')(input) # output shape : Batch Size X Filter Size X H_out X W_Out

        Part2 = tf.nn.conv2d(input, self.kernel, data_format='NCHW', padding='SAME') # output shape : Batch Size X Filter Size X H_out X W_Out

        # Tiling approach
        encoding_vector1 = tf.range(0, self.InputData_shape[2], 1, dtype=tf.float32)*self.scaleFactor # Defining a range from 0 to width of input tensor
        encoding_vector2 = tf.reshape(encoding_vector1,[1,1,1,-1]) # To Match with Input tensor shape


        Part3 = tf.multiply(Part2, encoding_vector2) # Element Wise Multiplication


        output =tf.math.add(Part1, Part3) # Adding the Conv2D output to fullfile the logic of EnKernel Approach

        return output

    def get_config(self):

        config = super(MyConv2D_TF, self).get_config()
        config.update({'scaleFactor': self.scaleFactor})
        return config

class MyConv2D(Layer): # Similar to EEGNet

    def __init__(self, cal_val, filter, output_dimension):
        self.filter = filter
        self.x = cal_val
        self.output_dimension = output_dimension
        # self.encoding_vector= encoding_vector
        super(MyConv2D, self).__init__()


    def build(self, input_shape):
        # Dimension of Kernel should be  F X C X H X W
        self.w = self.add_weight(name='kernel', shape=(int(self.filter), int(self.output_dimension[0]),
                                                       int(self.output_dimension[1]),
                                                       int(self.output_dimension[2])),
                                 initializer='glorot_uniform',
                                 regularizer='l1_l2',
                                 trainable=True)


        self.bs = 5  # shape(input_shape)[0] # BS
        C = 1  # input_shape[3] # Number of Channels
        self.H = 62  # input_shape[1] # Height of Input
        self.W = 375  # input_shape[2] # Width of Input
        self.w_tile = tf.tile(self.w, [1, 1, self.bs * self.H * self.W, 1])

        self.newKernel = self.w_tile
        # [Kf, Ch, Kh, Kw],
        super(MyConv2D, self).build(input_shape)

    def call(self, input):
        output = K.sum(self.newKernel * self.x, axis=3)

        return tf.reshape(output, [self.bs, self.filter, self.H, self.W])

    def get_output_shape_for(self, input_shape):

        return (self.bs, self.newH, self.newW,self.C )

class MyConv2D_Cleaner3(Layer): # With Leanrable Parameters


    def __init__(self, cal_val, filter,output_dimension):

        self.filter = filter
        self.x = cal_val
        self.output_dimension = output_dimension
        # self.encoding_vector= encoding_vector
        super(MyConv2D_Cleaner3,  self).__init__()

    # def get_config(self):
    #
    #     config = super().get_config().copy()
    #     config.update({
    #         'w': self.w,
    #         'scalefactor': self.scaleFactor,
    #     })
    #     return config
    
    def build(self, input_shape):

        # Dimension of Kernel should be  F X C X H X W
        self.w = self.add_weight(name='kernel', shape=(int(self.filter), int(self.output_dimension[0]),
                                                       int(self.output_dimension[1]),
                                                       int(self.output_dimension[2]) ),
                                 initializer='glorot_uniform',
                                 regularizer='l1_l2',
                                 trainable=True)


        self.scaleFactor =self.add_weight(shape=(1,), initializer="ones", trainable=True)

        self.bs = 5  # shape(input_shape)[0] # BS
        C = 1  # input_shape[3] # Number of Channels
        self.H = 62  # input_shape[1] # Height of Input
        self.W = 375  # input_shape[2] # Width of Input
        self.w_tile = tf.tile(self.w, [1, 1, self.bs * self.H * self.W, 1])

        # Tiling approach
        encoding_vector_step1 = tf.range(0, self.bs * self.H * self.W, 1, dtype=tf.float32)
        self.encoding_vector = tf.reshape(encoding_vector_step1 * self.scaleFactor,
                                           [1, 1, self.bs * self.H * self.W, 1])
        self.newKernel= self.w_tile + self.encoding_vector
         # [Kf, Ch, Kh, Kw],
        super(MyConv2D_Cleaner3, self).build(input_shape)

    def call(self, input):


        output=K.sum(self.newKernel * self.x,axis=3)

        return tf.reshape(output,[self.bs, self.filter, self.H, self.W])


class MyConv2D_Cleaner2(Layer): # Similar to EEGNet Layer


    def __init__(self, filter,output_dimension):

        self.filter = filter
        self.output_dimension = output_dimension
        super(MyConv2D_Cleaner2,  self).__init__()


    def build(self, input_shape):

        self.w_conv = self.add_weight(name='kernel', shape=(int(self.output_dimension[1]),
                                                       int(self.output_dimension[2]),
                                                       int(self.output_dimension[0]),int(self.filter) ),
                                 initializer='glorot_uniform',
                                 regularizer='l1_l2',
                                 trainable=True)

         # [Kf, Ch, Kh, Kw],
        super(MyConv2D_Cleaner2, self).build(input_shape)

    def call(self, input):


        output=conv(input, self.w_conv)

        return output

class MyConv2D_Cleaner(Layer):


    def __init__(self, cal_val, encoding_vector, filter,output_dimension, dynamic=True):
        #tf.reset_default_graph()
        self.filter = filter
        self.x = cal_val
        self.output_dimension = output_dimension
        self.encoding_vector= encoding_vector
        super(MyConv2D_Cleaner,  self).__init__()


    def build(self, input_shape):

        self.w = self.add_weight(name='kernel', shape=(int(self.output_dimension[0]),
                                                       int(self.output_dimension[1]),
                                                       int(self.output_dimension[2]), int(self.filter)),
                                 initializer='random_normal',
                                 trainable=True)

        # FLAG: No need to do this step , simply reshape will suffices
        kernel = []
        for j in range(self.filter):  # Filters
            kernel.append(tf.reshape(self.w[:, :, :, j], [1, -1]))
        self.w_reshape = tf.convert_to_tensor(kernel)
        # squares == [1, 4, 9, 16, 25, 36]

        super(MyConv2D_Cleaner, self).build(input_shape)

    def call(self, input):
        # with tf.Session() as sess:

        # input_shape=shape(input)

        self.bs=tf.shape(input)[0] # BS
        C =1# input_shape[3] # Number of Channels
        H =62# input_shape[1] # Height of Input
        W =375# input_shape[2] # Width of Input

        kernel_shape = shape(self.w)
        self.Kf=kernel_shape[3] # Number of Filter
        Kh= kernel_shape[1] # Height of Kernel
        Kw = kernel_shape[2] # Width of Kernel

        # If padding is SAME
        h_out = H
        w_out = W

        # if padding is VALID
        # h_out = (H - Kh + 2 * 0) / 1 + 1
        # w_out = (W - Kw + 2 * 0) / 1 + 1

        # print(W,' ',Kw,' ',w_out)
        # Based on Tiling approch
        # x_1_shape = shape(self.x)[1] # (H*W)
        # x_0_shape = shape(self.x)[0] # BS

        x_1_shape = tf.shape(self.x)[1] # (H*W)
        x_0_shape = tf.shape(self.x)[0] # BS

        # # Making sure dimension are integer or floor of input
        self.h_out, self.w_out = int(h_out), int(w_out)

        return looping_fast(self.x, self.w_reshape, 1, 0, self.Kf, C, Kh, Kw, self.bs, C, H, W, self.h_out,self.w_out, self.encoding_vector,x_1_shape,x_0_shape)


    def get_output_shape_for(self, input_shape):
        return (self.bs, self.Kf, self.h_out,self.w_out)  #[N, F, h_out,w_out]

def conv(ix, w):
   # filter shape: [filter_height, filter_width, in_channels, out_channels]
   # flatten filters
   filter_height = int(w.shape[0])
   filter_width = int(w.shape[1])
   in_channels = int(w.shape[2])
   out_channels = int(w.shape[3])
   ix_height = int(ix.shape[1])
   ix_width = int(ix.shape[2])
   ix_channels = int(ix.shape[3])
   filter_shape = [filter_height, filter_width, in_channels, out_channels]
   flat_w = tf.reshape(w, [filter_height * filter_width * in_channels, out_channels])
   patches = tf.extract_image_patches(
       ix,
       ksizes=[1, filter_height, filter_width, 1],
       strides=[1, 1, 1, 1],
       rates=[1, 1, 1, 1],
       padding='SAME'
   )
   patches_reshaped = tf.reshape(patches, [-1, ix_height, ix_width, filter_height * filter_width * ix_channels])
   feature_maps = []
   for i in range(out_channels):
       feature_map = tf.reduce_sum(tf.multiply(flat_w[:, i], patches_reshaped), axis=3, keep_dims=True)
       feature_maps.append(feature_map)
   features = tf.concat(feature_maps, axis=3)
   return features

def looping_fast (x, Kernel, stride, padding, F, kC, kH, kW, N, C, H, W, h_out, w_out,encoding_vector,x_1_shape,x_0_shape):

    # cnt = 0
    # ct = 0
    output_list = []
    # cnt=0

    # with tf.Session() as sess:
    # print(shape(x),' ',shape(Kernel))
    # for i in range(992):
    #     for j in range(124):
    #         output_list.append(
    #             K.sum(Kernel[j] * x[i],axis=1))


    # print(w_out)
    # Encoding Vector is creating column size increemental value of size kW followed by tiling of size h_out X w_out to match with
    # kernel_tield which is also equal to x_1_shape X kW. The x_1_shape = h_out X w_out

    for i in range(F):
        kernel_tiled = tf.reshape(tf.tile(tf.reshape(Kernel[i,:,:], [kW]), [x_1_shape]),[x_1_shape,kW])+ encoding_vector
        kernel_tiled_1 = tf.reshape(tf.tile(kernel_tiled,[1, x_0_shape]),[x_0_shape,x_1_shape,kW])
        output_list.append(
                    K.sum(kernel_tiled_1 * x,axis=2))



    return tf.reshape(tf.stack(output_list),[N, F, h_out,w_out])

# @tf.function
def looping (X, Kernel, stride, padding, F, kC, kH, kW, N, C, H, W):
    #tf.reset_default_graph()
    #tf.enable_eager_execution()
    cnt = 0
    ct = 0
    output_list = []
    h_out = (H - kH + 2 * padding) / stride + 1
    w_out = (W - kW + 2 * padding) / stride + 1


    # Checking if dimension are invalid
    if not h_out.is_integer() or not w_out.is_integer():
        raise Exception('Invalid output dimension!')

    # Making sure dimension are integer or floor of input
    h_out, w_out = int(h_out), int(w_out)
    #print(tf.shape(input))
    # with tf.Session() as sess:

        # for n in range(N):
        #     for f in range(F):
        #        for c in range(C):
        #            x_reshape=tf.reshape(X[n,c,:,:],[H,W])
        #            k_reshape=tf.reshape(Kernel[f,c,:,:],[kH,kW])
        #            for i in range(h_out):
        #                for j in range(w_out):
        #                    window = x_reshape[i: i + kH, j:j+kW]
        #                    output_list.append(K.sum(k_reshape * window))

    for n in range(N):
        for f in range(F):
            for c in range(C):
                x_reshape = tf.reshape(X[n, c, :, :], [H, W])
                k_reshape = tf.reshape(Kernel[f, c, :, :], [kH, kW])

                for i in range(h_out):
                    for j in range(w_out):
                        window = x_reshape[i: i + kH, j:j + kW]
                        output_list.append(K.sum(k_reshape * window))

    return tf.reshape(tf.stack(output_list),[N, F, h_out,w_out])
def body(X, Kernel, n, f, c, kH, kW, w_out, i,j):
    X_reshape=K.reshape(X[n, c, i: i + kH, j:j + kW], [1, -1])
    a=K.sum(
        Kernel * X_reshape)

    return a

def cond(X, Kernel,n, f, c, kH, Kw, w_out, i,j):
    return tf.less(j, w_out)
#@tf.function

def conv_new(X, W, stride, padding, n_filters, d_filter, h_filter, w_filter,n_x, d_x, h_x, w_x,sess ):

    # ref: https://wiseodd.github.io/techblog/2016/07/16/convnet-conv-layer/
    # Input X: DxCxHxW
    # Filter W: NFxCxHFxHW
    # Bias b: Fx1

    #cache = W, b, stride, padding
    # n_filters, d_filter, h_filter, w_filter = W.shape
    # n_x, d_x, h_x, w_x = X.shape
    h_out = (h_x - h_filter + 2 * padding) / stride + 1
    w_out = (w_x - w_filter + 2 * padding) / stride + 1
    # print(h_filter)
    # print(h_x)
    if not h_out.is_integer() or not w_out.is_integer():
        raise Exception('Invalid output dimension!')

    h_out, w_out = int(h_out),int(w_out)
    print(h_out,' ', w_out)

    X_col = im2col_indices(X, h_filter, w_filter, padding=padding, stride=stride)
    W_col = W.reshape(n_filters, -1) # Reshaping Kernel into N_filter x (Ch*H*W)


    # print('W',W_col.shape,' X',X_col.shape)
    out = W_col @ X_col #+ b  # Dot product between W_Col and X_Col

    # print('out', out.shape)
    out = out.reshape(n_filters, h_out, w_out, n_x)
    out = out.transpose(3, 0, 1, 2)

    # print(out.shape)
    # cache = (X, W, b, stride, padding, X_col)

    return tf.convert_to_tensor(out, dtype=tf.float32) #, cache

def conv_new_tf(X, W, stride, padding, n_filters, d_filter, h_filter, w_filter,n_x, d_x, h_x, w_x,sess ):

    # ref: https://wiseodd.github.io/techblog/2016/07/16/convnet-conv-layer/
    # Input X: DxCxHxW
    # Filter W: NFxCxHFxHW
    # Bias b: Fx1

    #cache = W, b, stride, padding
    # n_filters, d_filter, h_filter, w_filter = W.shape
    # n_x, d_x, h_x, w_x = X.shape
    h_out = (h_x - h_filter + 2 * padding) / stride + 1
    w_out = (w_x - w_filter + 2 * padding) / stride + 1
    # print(h_filter)
    # print(h_x)
    # if not h_out.is_integer() or not w_out.is_integer():
    #     raise Exception('Invalid output dimension!')

    h_out, w_out = tf.dtypes.cast(h_out, tf.int64), tf.dtypes.cast(w_out, tf.int64)

    X_col = im2col_indices_tf(X, h_filter, w_filter, padding=padding, stride=stride)
    # W_col = W.reshape(n_filters, -1) # Reshaping Kernel into N_filter x (Ch*H*W)
    W_col = tf.reshape(W,[n_filters, -1])  # Reshaping Kernel into N_filter x (Ch*H*W)

    # print('W',W_col.shape,' X',X_col.shape)
    # out = W_col @ X_col #+ b  # Dot product between W_Col and X_Col
    out = K.dot(W_col, X_col) #+ b  # Dot product between W_Col and X_Col
    # print('out', out.shape)
    # out = out.reshape(n_filters, h_out, w_out, n_x)
    # out = out.transpose(3, 0, 1, 2)
    out = tf.reshape(out, [n_filters, h_out, w_out, n_x])
    out = tf.transpose(out, perm=[3, 0, 1, 2])
    # print(out.shape)
    # cache = (X, W, b, stride, padding, X_col)

    # return tf.convert_to_tensor(out, dtype=tf.float32) #, cache
    return out#, cache

# New function base don tf.while loop - 30-Apr-2020
def conv_new2(X, Kernel, stride, padding, F, kC, kH, kW, N, C, H, W, sess):
    # Calculate dimension for output of convolution
    # It can move to main class for better computation
    h_out = (H - kH + 2 * padding) / stride + 1
    w_out = (W - kW + 2 * padding) / stride + 1

    i_N=i_F=i_h_out=i_w_out=i_C=tf.constant(0)
    # Checking if dimension are invalid
    if not h_out.is_integer() or not w_out.is_integer():
        raise Exception('Invalid output dimension!')

    # Making sure dimension are integer or floor of input
    h_out, w_out = int(h_out), int(w_out)
    #
    # def bodyN(X, Kernel, N, F, C, h_out, w_out,i_N,i_F,i_C,i_h_out,i_w_out):
    #     return tf.while_loop(condN, bodyF, [X, Kernel, N, F, C, h_out, w_out,i_N,i_F,i_C,i_h_out,i_w_out])



    def bodyF(X, Kernel, N, F, C, h_out, w_out,i_N,i_F,i_C,i_h_out,i_w_out):
        return tf.while_loop(condF, bodyC, [X, Kernel, N, F, C, h_out, w_out,i_N,i_F,i_C,i_h_out,i_w_out])

    def bodyC(X, Kernel, N, F, C, h_out, w_out,i_N,i_F,i_C,i_h_out,i_w_out):
        return tf.while_loop(condC, body_h_out, [X, Kernel, N, F, C, h_out, w_out,i_N,i_F,i_C,i_h_out,i_w_out])

    def body_h_out(X, Kernel, N, F, C, h_out, w_out,i_N,i_F,i_C,i_h_out,i_w_out):
        return tf.while_loop(cond_h_out, body_w_out, [X, Kernel, N, F, C, h_out, w_out,i_N,i_F,i_C,i_h_out,i_w_out])

    def body_w_out(X, Kernel, N, F, C, h_out, w_out,i_N,i_F,i_C,i_h_out,i_w_out):
        return tf.while_loop(cond_w_out, body_main, [X, Kernel, N, F, C, h_out, w_out,i_N,i_F,i_C,i_h_out,i_w_out])

    def body_main(X, Kernel, N, F, C, h_out, w_out,i_N,i_F,i_C,i_h_out,i_w_out):
        output = K.sum(K.dot(Kernel[F, C, :, :] , X[N, C, h_out:h_out + i_h_out, w_out:w_out + i_w_out]))
        print(tf.shape(output))
        return output

    def condN(X, Kernel, N, F, C, h_out, w_out,i_N,i_F,i_C,i_h_out,i_w_out):
        return tf.less(i_N,N)

    def condF(X, Kernel, N, F, C, h_out, w_out,i_N,i_F,i_C,i_h_out,i_w_out):
        return tf.less(i_F,F)

    def condC(X, Kernel, N, F, C, h_out, w_out,i_N,i_F,i_C,i_h_out,i_w_out):
        return tf.less(i_C,C)

    def cond_h_out(X, Kernel, N, F, C, h_out, w_out,i_N,i_F,i_C,i_h_out,i_w_out):
        return tf.less(i_h_out,h_out)

    def cond_w_out(X, Kernel, N, F, C, h_out, w_out,i_N,i_F,i_C,i_h_out,i_w_out):
        return tf.less(i_w_out,w_out)

    return tf.while_loop(condN, bodyF, [X, Kernel, N, F, C, h_out, w_out, i_N, i_F, i_C, i_h_out, i_w_out])

def convol(images,weights,biases,stride):
    """
    Args:
      images:input images or features, 4-D tensor
      weights:weights, 4-D tensor
      biases:biases, 1-D tensor
      stride:stride, a float number
    Returns:
      conv_feature: convolved feature map
    """
    image_num = images.shape[0] #the number of input images or feature maps
    channel = images.shape[1] #channels of an image,images's shape should be like [n,c,h,w]
    weight_num = weights.shape[0] #number of weights, weights' shape should be like [n,c,size,size]
    ksize = weights.shape[2]
    h = images.shape[2]
    w = images.shape[3]
    out_h = (h+np.floor(ksize/2)*2-ksize)/2+1
    out_w = out_h

    conv_features = np.zeros([image_num,weight_num,out_h,out_w])
    for i in range(image_num):
        image = images[i,...,...,...]
        for j in range(weight_num):
            sum_convol_feature = np.zeros([out_h,out_w])
            for c in range(channel):
                #extract a single channel image
                channel_image = image[c,...,...]
                #pad the image
                padded_image = im_pad(channel_image,ksize/2)
                #transform this image to a vector
                im_col = im2col(padded_image,ksize,stride)

                weight = weights[j,c,...,...]
                weight_col = np.reshape(weight,[-1])
                mul = np.dot(im_col,weight_col)
                convol_feature = np.reshape(mul,[out_h,out_w])
                sum_convol_feature = sum_convol_feature + convol_feature
            conv_features[i,j,...,...] = sum_convol_feature + biases[j]
    return conv_features





































# Convolution Operatio

def convolve2d(imagee, kernell):


    """
    This function which takes an image and a kernel and returns the convolution of them.
    :param image: a numpy array of size [image_height, image_width].
    :param kernel: a numpy array of size [kernel_height, kernel_width].
    :return: a numpy array of size [image_height, image_width] (convolution output).
    """
    # Flip the kernel
    # kernel = np.flipud(np.fliplr(kernel))
    # convolution output
    image=imagee#.eval(session=sess)
    kernel=kernell#.eval(session=sess)
    H =image.shape[1]
    W = image.shape[2]
    C = image.shape[3]
    Kh, Kw, Kc = kernel.shape
    C=Kc=1
    newH = H - Kh + 1
    newW = W - Kw + 1
    output = np.zeros((newH, newW, C))
    print((image.numpy()).shape)
    cnt=0
    ct=0
    print((image.numpy()).shape)
    for chan_size in range(C):
        for y in range(newW):
            cnt = cnt + 0
            # print(cnt)
            for x in range(newH):
                # print('kernel = ', tf.shape(kernel[:, :, chan_size]), ' image = ', tf.shape(image[x: x + Kh,
                #                                                                y: y + Kw,
                #                                                                chan_size]))
                ct = ct + 10

                # output[x, y, chan_size] = ((kernel[:, :, chan_size] + (
                #             kernel[:, :, chan_size] * (cnt)) + (kernel[:, :, chan_size] * (ct))) * image[
                #                                                                                                 x: x +
                #                                                                                                    kernel.shape[
                #                                                                                                        0],
                #                                                                                                 y: y +
                #                                                                                                    kernel.shape[
                #                                                                                                        1],
                #                                                                                                 chan_size]).sum()

                output[x, y, chan_size] = (( (
                        kernel[:, :, chan_size] * (cnt)) + (kernel[:, :, chan_size] * (ct))) * image[
                                                                                               x: x +
                                                                                                  Kh,
                                                                                               y: y +
                                                                                                  Kw,
                                                                                               chan_size]).sum()
            ct=0
        cnt=0


    return output










######
np.random.seed(1)


def tf_int_round(num):
	return tf.cast(tf.round(num), dtype=tf.int32)
# Defining Zero Padding

def zero_pad(X, pad):
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=(pad, pad))

    return X_pad

# Single step convolution
def conv_single_step(a_slice_prev, W, b):
    # Element-wise product between a_slice and W. Do not add the bias yet.
    s = np.multiply(a_slice_prev, W)

    # Sum over all entries of the volume s.
    Z = np.sum(s)

    # Add bias b to Z. Cast b to a float() so that Z results in a scalar value.
    Z = Z + float(b)

    return Z

# Convolution Forward
def conv2d_multi_channel(input, w):
    """Two-dimensional convolution with multiple channels.

    Uses SAME padding with 0s, a stride of 1 and no dilation.

    input: input array with shape (height, width, in_depth)
    w: filter array with shape (fd, fd, in_depth, out_depth) with odd fd.
       in_depth is the number of input channels, and has the be the same as
       input's in_depth; out_depth is the number of output channels.

    Returns a result with shape (height, width, out_depth).
    """
    #assert w.shape[0] == w.shape[1] and w.shape[0] % 2 == 1

    # padw = w.shape[0] // 2
    # padded_input = np.pad(input,
    #                       pad_width=((padw, padw), (padw, padw), (0, 0)),
    #                       mode='constant',
    #                       constant_values=0)

    print("Input shape ", input.shape)
    print("Kernel shape ",  w.shape)

    height = input.shape[1]
    width = input.shape[2]
    in_depth = input.shape[3]
    # assert in_depth == w.shape[2]
    out_depth = w.shape[3]
    #output = np.zeros((16,height, width, out_depth)) # batch size should be 16 as per training
    output = np.zeros(input.shape) # batch size should be 16 as per training






    for bs in range(input.shape[0]):  # batch size should be 16 as per training
        for out_c in range(out_depth):
            # For each output channel, perform 2d convolution summed across all
            # input channels.
            for i in range(height):
                for j in range(width):
                    # Now the inner loop also works across all input channels.
                    for c in range(in_depth):
                        for fi in range(w.shape[0]):
                            for fj in range(w.shape[1]):
                                w_element = w[fi, fj, c, out_c]
                                #print(input[bs,i + fi, j + fj, c])
                                output[bs,i, j, out_c] += (input[bs,i + fi, j + fj, c] * w_element)
    return output

def conv2d_single_channel(input, w):
    """Two-dimensional convolution of a single channel.

    Uses SAME padding with 0s, a stride of 1 and no dilation.

    input: input array with shape (height, width)
    w: filter array with shape (fd, fd) with odd fd.

    Returns a result with the same shape as input.
    """
    assert w.shape[0] == w.shape[1] and w.shape[0] % 2 == 1

    # SAME padding with zeros: creating a new padded array to simplify index
    # calculations and to avoid checking boundary conditions in the inner loop.
    # padded_input is like input, but padded on all sides with
    # half-the-filter-width of zeros.
    padded_input = np.pad(input,
                          pad_width=w.shape[0] // 2,
                          mode='constant',
                          constant_values=0)

    output = np.zeros_like(input)
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            # This inner double loop computes every output element, by
            # multiplying the corresponding window into the input with the
            # filter.
            for fi in range(w.shape[0]):
                for fj in range(w.shape[1]):
                    output[i, j] += padded_input[i + fi, j + fj] * w[fi, fj]
    return output

def conv_forward(A_prev, W, b, hparameters):
    """
    Implements the forward propagation for a convolution function
    Arguments:
    A_prev -- output activations of the previous layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    hparameters -- python dictionary containing "stride" and "pad"
    Returns:
    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward() function
    """

    ### START CODE HERE ###
    # Retrieve dimensions from A_prev's shape (≈1 line)
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # Retrieve dimensions from W's shape (≈1 line)
    (f, f, n_C_prev, n_C) = W.shape

    # Retrieve information from "hparameters" (≈2 lines)
    stride = hparameters['stride']
    pad = hparameters['pad']

    # Compute the dimensions of the CONV output volume using the formula given above. Hint: use int() to floor. (≈2 lines)
    n_H = int((n_H_prev - f + (2 * pad)) / stride + 1)
    n_W = int((n_W_prev - f + (2 * pad)) / stride + 1)

    # Initialize the output volume Z with zeros. (≈1 line)
    Z = np.zeros((m, n_H, n_W, n_C))

    # Create A_prev_pad by padding A_prev
    A_prev_pad = zero_pad(A_prev, pad)

    for i in range(m):  # loop over the batch of training examples
        a_prev_pad = A_prev[i]  # Select ith training example's padded activation
        for h in range(n_H):  # loop over vertical axis of the output volume
            for w in range(n_W):  # loop over horizontal axis of the output volume
                for c in range(n_C):  # loop over channels (= #filters) of the output volume

                    # Find the corners of the current "slice" (≈4 lines)
                    vert_start = stride * h
                    vert_end = stride * h + f
                    horiz_start = stride * w
                    horiz_end = stride * w + f

                    # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell). (≈1 line)
                    a_slice_prev = A_prev_pad[i, vert_start:vert_end, horiz_start:horiz_end, :]

                    # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron. (≈1 line)
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, W[:, :, :, c], b[:, :, :, c])

    ### END CODE HERE ###

    # Making sure your output shape is correct
    assert (Z.shape == (m, n_H, n_W, n_C))

    # Save information in "cache" for the backprop
    cache = (A_prev, W, b, hparameters)

    return Z, cache

# Pooling
def pool_forward(A_prev, hparameters, mode="max"):
    # Retrieve dimensions from the input shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # Retrieve hyperparameters from "hparameters"
    f = hparameters["f"]
    stride = hparameters["stride"]

    # Define the dimensions of the output
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev

    # Initialize output matrix A
    A = np.zeros((m, n_H, n_W, n_C))

    for i in range(m):  # loop over the training examples
        for h in range(n_H):  # loop on the vertical axis of the output volume
            for w in range(n_W):  # loop on the horizontal axis of the output volume
                for c in range(n_C):  # loop over the channels of the output volume

                    # Find the corners of the current "slice"
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    # Use the corners to define the current slice on the ith training example of A_prev, channel c. (≈1 line)
                    a_prev_slice = A_prev[vert_start:vert_end, horiz_start:horiz_end, :]

                    # Compute the pooling operation on the slice. Use an if statment to differentiate the modes. Use np.max/np.mean.
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_prev_slice)

    # Store the input and hparameters in "cache" for pool_backward()
    cache = (A_prev, hparameters)

    # Making sure your output shape is correct
    assert (A.shape == (m, n_H, n_W, n_C))

    return A, cache

# Convolution Backword
def conv_backward(dZ, cache):
    # Retrieve information from "cache"
    (A_prev, W, b, hparameters) = cache

    # Retrieve dimensions from A_prev's shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # Retrieve dimensions from W's shape
    (f, f, n_C_prev, n_C) = W.shape

    # Retrieve information from "hparameters"
    stride = hparameters["stride"]
    pad = hparameters["pad"]

    # Retrieve dimensions from dZ's shape
    (m, n_H, n_W, n_C) = dZ.shape

    # Initialize dA_prev, dW, db with the correct shapes
    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
    dW = np.zeros((f, f, n_C_prev, n_C))
    db = np.zeros((1, 1, 1, n_C))

    # Pad A_prev and dA_prev
    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)

    for i in range(m):  # loop over the training examples

        # select ith training example from A_prev_pad and dA_prev_pad
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]

        for h in range(n_H):  # loop over vertical axis of the output volume
            for w in range(n_W):  # loop over horizontal axis of the output volume
                for c in range(n_C):  # loop over the channels of the output volume

                    # Find the corners of the current "slice"
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    # Use the corners to define the slice from a_prev_pad
                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                    # Update gradients for the window and the filter's parameters
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]

        # Set the ith training example's dA_prev to the unpaded da_prev_pad
        dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]

    # Making sure your output shape is correct
    assert (dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))

    return dA_prev, dW, db

# max Pooling
def create_mask_from_window(x):
    mask = x == np.max(x)

    return mask

# Average Pooling
def distribute_value(dz, shape):
    # Retrieve dimensions from shape
    (n_H, n_W) = shape

    # Compute the value to distribute on the matrix
    average = dz / (n_H * n_W)

    # Create a matrix where every entry is the "average" value
    a = np.ones(shape) * average

    return a


def pool_backward(dA, cache, mode="max"):
    # Retrieve information from cache
    (A_prev, hparameters) = cache

    # Retrieve hyperparameters from "hparameters"
    stride = hparameters['stride']
    f = hparameters['f']

    # Retrieve dimensions from A_prev's shape and dA's shape
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    m, n_H, n_W, n_C = dA.shape

    # Initialize dA_prev with zeros
    dA_prev = np.zeros(A_prev.shape)

    for i in range(m):  # loop over the training examples

        # select training example from A_prev
        a_prev = A_prev[i]

        for h in range(n_H):  # loop on the vertical axis
            for w in range(n_W):  # loop on the horizontal axis
                for c in range(n_C):  # loop over the channels (depth)

                    # Find the corners of the current "slice"
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    # Compute the backward propagation in both modes.
                    if mode == "max":

                        # Use the corners and "c" to define the current slice from a_prev
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]

                        # Create the mask from a_prev_slice (≈1 line)
                        mask = create_mask_from_window(a_prev_slice)

                        # Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA)
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += np.multiply(mask, dA[i, h, w, c])

                    elif mode == "average":

                        # Get the value a from dA
                        da = dA[i, h, w, c]

                        # Define the shape of the filter as fxf
                        shape = (f, f)

                        # Distribute it to get the correct slice of dA_prev. i.e. Add the distributed value of da
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += distribute_value(da, shape)

    # Making sure your output shape is correct
    assert (dA_prev.shape == A_prev.shape)

    return dA_prev

# # Testing Zero-pad
# np.random.seed(1)
# x = np.random.randn(4, 3, 3, 2)
# x_pad = zero_pad(x, 2)
# print ("x.shape =", x.shape)
# print ("x_pad.shape =", x_pad.shape)
# print ("x[1,1] =", x[1,1])
# print ("x_pad[1,1] =", x_pad[1,1])
#
# # Testing Single step convolution
# fig, axarr = plt.subplots(1, 2)
# axarr[0].set_title('x')
# axarr[0].imshow(x[0,:,:,0])
# axarr[1].set_title('x_pad')
# axarr[1].imshow(x_pad[0,:,:,0])
#
# np.random.seed(1)
# a_slice_prev = np.random.randn(4, 4, 3)
# W = np.random.randn(4, 4, 3)
# b = np.random.randn(1, 1, 1)
# Z = conv_single_step(a_slice_prev, W, b)
# print("Z =", Z)
#
# # Testing Convolution Forward
# np.random.seed(1)
# A_prev = np.random.randn(10,4,4,3)
# W = np.random.randn(2,2,3,8)
# b = np.random.randn(1,1,1,8)
# hparameters = {"pad" : 2,
#                "stride": 2}
#
# Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
# print("Z's mean =", np.mean(Z))
# print("Z[3,2,1] =", Z[3,2,1])
# print("cache_conv[0][1][2][3] =", cache_conv[0][1][2][3])
#
# # testing Pooling forward
# np.random.seed(1)
# A_prev = np.random.randn(2, 4, 4, 3)
# hparameters = {"stride" : 2, "f": 3}
#
# A, cache = pool_forward(A_prev, hparameters)
# print("mode = max")
# print("A =", A)
# print()
# A, cache = pool_forward(A_prev, hparameters, mode = "average")
# print("mode = average")
# print("A =", A)
#
#
# # Testing Backword Convolution
# np.random.seed(1)
# dA, dW, db = conv_backward(Z, cache_conv)
# print("dA_mean =", np.mean(dA))
# print("dW_mean =", np.mean(dW))
# print("db_mean =", np.mean(db))
#
# # testing max-pooling
# x = np.random.randn(2,3)
# mask = create_mask_from_window(x)
# print('x = ', x)
# print("mask = ", mask)
#
# # testing Avg-pooling
# a = distribute_value(2, (2,2))
# print('distributed value =', a)
