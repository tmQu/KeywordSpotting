import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, BatchNormalization, ReLU, AveragePooling2D, Flatten, Dense, Input,Reshape, GaussianNoise
from tensorflow.keras.models import Model

# https://arxiv.org/pdf/1704.04861
# https://github.com/haoheliu/Key-word-spotting-DNN-GRU-DSCNN/blob/master/model.py
class DS_CNN:
    def __init__(self, input_shape, num_classes, model_size_info = [3, 16, 10, 4, 2, 1, 16, 3, 3, 2, 2, 16, 3, 3, 1, 1]):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_size_info = model_size_info

    def build_model(self):
        def _depthwise_separable_conv(inputs, num_pwc_filters, kernel_size, stride, name):
            x = DepthwiseConv2D(kernel_size=kernel_size, strides=stride, padding='same', use_bias=False, name=name + '_dw')(inputs)
            x = BatchNormalization(name=name + '_dw_bn')(x)
            x = ReLU(name=name + '_dw_relu')(x)
            x = Conv2D(num_pwc_filters, kernel_size=(1, 1), padding='same', use_bias=False, name=name + '_pw')(x)
            x = BatchNormalization(name=name + '_pw_bn')(x)
            x = ReLU(name=name + '_pw_relu')(x)
            return x
    
        channels = 1
        columns = 13
        rows = int(self.input_shape / (columns * channels))
        
        input_tensor = Input(shape=(self.input_shape))
        x = Reshape((rows, columns, channels))(input_tensor)
        x = GaussianNoise(stddev=0.45)(x)
        
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
                x = Conv2D(conv_feat[layer_no], kernel_size=(conv_kt[layer_no], conv_kf[layer_no]), strides=(conv_st[layer_no], conv_sf[layer_no]), padding='same', use_bias=False, name='conv_1')(x)
                x = BatchNormalization(name='conv_1_bn')(x)
                x = ReLU(name='conv_1_relu')(x)
            else:
                x = _depthwise_separable_conv(x, conv_feat[layer_no], kernel_size=(conv_kt[layer_no], conv_kf[layer_no]), stride=(conv_st[layer_no], conv_sf[layer_no]), name='conv_ds_' + str(layer_no))
        
        t_dim = rows // conv_st[0]
        f_dim = columns // conv_sf[0]
        for layer_no in range(1, num_layers):
            t_dim = t_dim // conv_st[layer_no]
            f_dim = f_dim // conv_sf[layer_no]
        
        x = AveragePooling2D(pool_size=(t_dim, f_dim), name='avg_pool')(x)
        x = Flatten(name='flatten')(x)
        output_tensor = Dense(self.num_classes, activation='softmax', name='fc1')(x)
        
        model = Model(inputs=input_tensor, outputs=output_tensor)
        return model




if __name__ == '__main__':
    model = DS_CNN((650), 3)
    model = model.build_model()
    model.summary()

