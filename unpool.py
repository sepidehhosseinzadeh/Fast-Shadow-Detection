from lasagne import layers
import numpy as np

class Unpool2DLayer(layers.Layer):
    """
    This layer performs unpooling over the last two dimensions
    of a 4D tensor.
    """
    def __init__(self, incoming, ds, **kwargs):

        super(Unpool2DLayer, self).__init__(incoming, **kwargs);

        if (isinstance(ds, int)):
            raise ValueError('ds must have len == 2');
        else:
            ds = tuple(ds);
            if len(ds) != 2:
                raise ValueError('ds must have len == 2');
            if ds[0] != ds[1]:
                raise ValueError('ds should be symmetric (I am lazy)');
            self.ds = ds;

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape);

        output_shape[2] = input_shape[2] * self.ds[0];
        output_shape[3] = input_shape[3] * self.ds[1];

        return tuple(output_shape);

    def get_output_for(self, input, **kwargs):
        ds = self.ds;
        input_shape = input.shape;
        output_shape = self.get_output_shape_for(input_shape);
        return input.repeat(2, axis=2).repeat(2, axis=3);

