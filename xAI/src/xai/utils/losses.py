'''
Predefined loss functions for analyzer methods.

Expected input data shape: (batch_size, outputs)
'''

from tensorflow.keras import backend as K

mean_loss = lambda x, axis=None: K.mean(x, axis=axis, keepdims=True)
max_loss = lambda x, axis=None: K.max(x, axis=axis, keepdims=True)
index_loss = lambda x, index: x[:, index] # not relevant for our case
identity_loss = lambda x, axis=None: x # unused `axis` argument to match compatibility
