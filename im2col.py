import numpy as np
import tensorflow.compat.v1 as tf

def shape(tensor):
    s = tensor.get_shape()
    return tuple([s[i].value for i in range(0, len(s))])

def get_im2col_indices_tf(x_shape, field_height, field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_width) % stride == 0
    out_height = int((H + 2 * padding - field_height) / stride + 1) # Formula to calculate Dim of Conv Matrix
    out_width = int((W + 2 * padding - field_width) / stride + 1)

    print('FH',field_height,'FW',field_width)
    i0 = np.repeat(np.arange(field_height), field_width)
    print('arr in FH = ', np.arange(field_height))
    print('repeat in FW = ', i0)
    i0 = np.tile(i0, C)
    # print('C = ', C, ' Tile in C = ',i0)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (np.int32(k), np.int32(i), np.int32(j))


def im2col_indices_tf(x, field_height, field_width, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    padding=1
    p = padding
    x_padded =tf.pad(x,[[0, 0], [0, 0], [p, p], [p, p]],"CONSTANT") #np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    # k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)
    k, i, j = get_im2col_indices_tf(shape(x), field_height, field_width, padding, stride)


    print('i j k fh fw x_padded', i.shape,' ',j.shape,' ',k.shape,' ',field_height, ' ',field_width, ' ', shape(x_padded))
    # cols = x_padded[:, k, i, j]
    cols = x_padded[...,[k.shape[0], k.shape[1]], [i.shape[0], i.shape[1]], [j.shape[0], j.shape[1]]]
    print(shape(cols))
    # C = x.shape[1]
    C = shape(x)[1]
    # cols=cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    cols = tf.reshape(tf.transpose(cols, perm=[1, 2, 0]),[field_height * field_width * C, -1])
    return cols

def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_width) % stride == 0
    out_height =int( (H + 2 * padding - field_height) / stride + 1)
    out_width =int( (W + 2 * padding - field_width) / stride + 1)
    print(field_width, ' ', field_height)
    print(out_width,' ', out_height)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k, i, j)

def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                                 stride)

    print('i j k fh fw x_padded', i.shape, ' ', j.shape, ' ', k.shape, ' ', field_height, ' ', field_width, ' ',
          x_padded.shape)
    # cols = x_padded[:, k, i, j]

    cols = x_padded[:, k, i, j]
    print(cols.shape)
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols

def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1,
                   stride=1):
    """ An implementation of col2im based on fancy indexing and np.add.at """
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]
def im2col_tf(x,hh,ww,stride):

    """
    Args:
      x: image matrix to be translated into columns, (C,H,W)
      hh: filter height
      ww: filter width
      stride: stride
    Returns:
      col: (new_h*new_w,hh*ww*C) matrix, each column is a cube that will convolve with a filter
            new_h = (H-hh) // stride + 1, new_w = (W-ww) // stride + 1
    """

    c,h,w = x.shape
    new_h = (h-hh) // stride + 1
    new_w = (w-ww) // stride + 1
    col = tf.zeros([new_h*new_w,c*hh*ww])

    for i in range(new_h):
       for j in range(new_w):
           patch = x[:,i*stride:i*stride+hh,j*stride:j*stride+ww]
           col[i*new_w+j,:] = tf.reshape(patch,-1)
    return col


def im2col(x,hh,ww,stride):

    """
    Args:
      x: image matrix to be translated into columns, (C,H,W)
      hh: filter height
      ww: filter width
      stride: stride
    Returns:
      col: (new_h*new_w,hh*ww*C) matrix, each column is a cube that will convolve with a filter
            new_h = (H-hh) // stride + 1, new_w = (W-ww) // stride + 1
    """

    c,h,w = x.shape
    new_h = (h-hh) // stride + 1
    new_w = (w-ww) // stride + 1
    col = np.zeros([new_h*new_w,c*hh*ww])

    for i in range(new_h):
       for j in range(new_w):
           patch = x[...,i*stride:i*stride+hh,j*stride:j*stride+ww]
           col[i*new_w+j,:] = np.reshape(patch,-1)
    return col

def col2im(mul,h_prime,w_prime,C):
    """
      Args:
      mul: (h_prime*w_prime*w,F) matrix, each col should be reshaped to C*h_prime*w_prime when C>0, or h_prime*w_prime when C = 0
      h_prime: reshaped filter height
      w_prime: reshaped filter width
      C: reshaped filter channel, if 0, reshape the filter to 2D, Otherwise reshape it to 3D
    Returns:
      if C == 0: (F,h_prime,w_prime) matrix
      Otherwise: (F,C,h_prime,w_prime) matrix
    """
    F = mul.shape[1]
    if(C == 1):
        out = np.zeros([F,h_prime,w_prime])
        for i in range(F):
            col = mul[:,i]
            out[i,:,:] = np.reshape(col,(h_prime,w_prime))
    else:
        out = np.zeros([F,C,h_prime,w_prime])
        for i in range(F):
            col = mul[:,i]
            out[i,:,:] = np.reshape(col,(C,h_prime,w_prime))

    return out