
from kws_streaming.layers import modes
from kws_streaming.layers import speech_features
from kws_streaming.layers.compat import tf
from tensorflow.python.ops import gen_audio_ops as audio_ops  # p

class MyReshape(tf.keras.layers.Layer):
    def __init__(self, num_splits=26, target_shape = None, **kwargs):
        super(MyReshape, self).__init__(**kwargs)
        self.num_splits = int(num_splits)
        self.target_shape = target_shape


    def build(self, input_shape):
        super(MyReshape, self).build(input_shape)
        self.channels = 1
        self.columns = self.num_splits
        self.rows = input_shape[1] // self.num_splits
        self.original_shape = input_shape[1]
        self.target_shape = (self.rows, self.columns, 1)
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        return tf.reshape(inputs, [batch_size, self.rows, self.columns, 1])

    def get_config(self):
        base_config = super(MyReshape, self).get_config()
        config = {
            'num_splits': self.num_splits,
            'target_shape': (self.rows, self.columns, 1)
        }
        return dict(list(base_config.items()) + list(config.items()))

