
from kws_streaming.layers import modes
from kws_streaming.layers import stream
from kws_streaming.layers.compat import tf
# from kws_streaming.models.utils import parse

# https://arxiv.org/pdf/1704.04861
# https://github.com/haoheliu/Key-word-spotting-DNN-GRU-DSCNN/blob/master/model.py
from model.MyReshape import MyReshape

class DS_CNN:
    def __init__(self, input_shape, num_classes,  model_size_info = [3, 64, 1, 1, 6, 6, 64, 1, 1, 2, 2, 64, 1, 1, 1, 1], reshape = 65 ):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_size_info = model_size_info
        self.reshape = reshape
    def build_model2(self):
        input_tensor = tf.keras.layers.Input(shape=(self.input_shape))
        # x = MyReshape(13)(input_tensor)
        
        x = tf.keras.backend.expand_dims(input_tensor, 2, -2)
        # x = tf.keras.backend.expand_dims(x, -3)

        model = tf.keras.Model(inputs=input_tensor, outputs=x)
        return model
    def build_model(self):
        def _depthwise_separable_conv(inputs, num_pwc_filters, kernel_size, stride):
            x = stream.Stream(tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size, strides=stride, padding='valid', use_bias=False))(inputs)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            
            
            x = tf.keras.layers.Conv2D(num_pwc_filters, kernel_size=(1, 1), padding='valid', use_bias=False)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            return x
    
        channels = 1
        columns = self.reshape #
        rows = self.input_shape // columns

        input_tensor = tf.keras.layers.Input(shape=(self.input_shape))
        x = MyReshape(columns)(input_tensor)
        # x = tf.keras.layers.Reshape((rows, columns, 1))(input_tensor)
        # x = tf.keras.backend.expand_dims(input_tensor, -3)
        # x = tf.keras.backend.expand_dims(x, -3)
        # x = tf.keras.backend.expand_dims(x)
        # # x = tf.keras.backend.expand_dims(x)
        num_layers = self.model_size_info[0]
        conv_feat = [None] * num_layers
        conv_kt = [None] * num_layers  # Conv filter height
        conv_st = [None] * num_layers  # Conv filter stride
        conv_kf = [None] * num_layers  # Conv filter width
        conv_sf = [None] * num_layers  # Conv filter stride width
        
        i = 1
        for layer_no in range(num_layers):
            conv_feat[layer_no] = self.model_size_info[i]
            i += 1
            conv_kt[layer_no] = self.model_size_info[i]
            i += 1
            conv_kf[layer_no] = self.model_size_info[i]
            i += 1
            conv_st[layer_no] = self.model_size_info[i]
            i += 1
            conv_sf[layer_no] = self.model_size_info[i]
            i += 1


        for layer_no in range(num_layers):
            if layer_no == 0:
                x = stream.Stream(tf.keras.layers.Conv2D(conv_feat[layer_no], kernel_size=(conv_kt[layer_no], conv_kf[layer_no]), strides=(conv_st[layer_no], conv_sf[layer_no]), padding='valid', use_bias=False))(x)
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.Activation('relu')(x)
            else:
                x = _depthwise_separable_conv(x, conv_feat[layer_no], kernel_size=(conv_kt[layer_no], conv_kf[layer_no]), stride=(conv_st[layer_no], conv_sf[layer_no]))
        
        t_dim = rows // conv_st[0]
        f_dim = columns // conv_sf[0]
        for layer_no in range(1, num_layers):
            t_dim = t_dim // conv_st[layer_no]
            f_dim = f_dim // conv_sf[layer_no]
        
        x = stream.Stream(tf.keras.layers.AveragePooling2D(pool_size=(int(x.shape[1]), int(x.shape[2]) )))(x)
        x = stream.Stream(tf.keras.layers.Flatten())(x)
        x = tf.keras.layers.Dense(self.num_classes)(x)
        output_tensor = tf.keras.layers.Activation('softmax')(x)
        model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
        return model




if __name__ == '__main__':
    model = DS_CNN((650), 3)
    model = model.build_model()
    model.summary()

