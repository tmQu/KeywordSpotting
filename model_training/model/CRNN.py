from tensorflow.keras.layers import Input, Reshape, Conv2D, BatchNormalization, ReLU, Dropout, Bidirectional, GRU, Dense, GaussianNoise, Flatten, LSTM, concatenate
from tensorflow.keras.models import Model


# 
# class CRNN:
#     def __init__(self, input_shape, num_classes, 
#                  model_size_info= [16, 3, 3, 1, 1, 1, 32, 8], 
#                  is_training=True):
#         self.input_shape = input_shape
#         self.num_classes = num_classes
#         self.model_size_info = model_size_info
#         print(self.model_size_info)
#         self.is_training = is_training

#     def build_model(self):
#         channels = 1
#         columns = 13
#         rows = int(self.input_shape / (columns * channels))
        
#         input_tensor = Input(shape=(self.input_shape,))
#         x = Reshape((rows, columns, channels))(input_tensor)
#         x = GaussianNoise(stddev=0.45)(x)

#         # CNN part
#         first_filter_count = self.model_size_info[0]
#         first_filter_height = self.model_size_info[1]
#         first_filter_width = self.model_size_info[2]
#         first_filter_stride_y = self.model_size_info[3]
#         first_filter_stride_x = self.model_size_info[4]

#         x = Conv2D(first_filter_count, 
#                    (first_filter_height, first_filter_width), 
#                    strides=(first_filter_stride_y, first_filter_stride_x), 
#                    padding='valid', 
#                    kernel_initializer='glorot_uniform', 
#                    name='conv_1')(x)
#         x = BatchNormalization()(x)
#         x = ReLU()(x)
#         if self.is_training:
#             x = Dropout(0.5)(x)

#         first_conv_output_height = (rows - first_filter_height + first_filter_stride_y) // first_filter_stride_y
#         first_conv_output_width = (columns - first_filter_width + first_filter_stride_x) // first_filter_stride_x

#         # Reshape for GRU part
#         x = Reshape((first_conv_output_height, first_conv_output_width * first_filter_count))(x)

#         # GRU part
#         num_rnn_layers = self.model_size_info[5]
#         RNN_units = self.model_size_info[6]
#         print(RNN_units)
        
#         for i in range(num_rnn_layers):
#             if i == 0:
#                 if self.is_training:
#                     x = Bidi    rectional(LSTM(RNN_units, return_sequences=True, kernel_initializer='glorot_uniform', unroll =False), name=f'lstm_{i+1}')(x)
#                 else:
#                     x = Bidirectional(LSTM(RNN_units, return_sequences=False, kernel_initializer='glorot_uniform', unroll =False), name=f'lstm_{i+1}')(x)
#             else:
#                 x = Bidirectional(LSTM(RNN_units, return_sequences=False, kernel_initializer='glorot_uniform', unroll =False), name=f'lstm_{i+1}')(x)
        
#         # Fully Connected part
#         x = Flatten()(x)

#         first_fc_output_channels = self.model_size_info[7]
#         x = Dense(first_fc_output_channels, activation='relu', kernel_initializer='glorot_uniform', name='fc1')(x)
#         if self.is_training:
#             x = Dropout(0.5)(x)

#         # Output layer
#         output_tensor = Dense(self.num_classes, activation='softmax', kernel_initializer='glorot_uniform', name='output')(x)
        
#         model = Model(inputs=input_tensor, outputs=output_tensor)
#         return model


# this model is based on https://github.com/ARM-software/ML-KWS-for-MCU/blob/master/models.py#L963
# this is the variant to work well with the Edge Impulse especially for the Edge devices

class CRNN:
    def __init__(self, input_shape, num_classes, 
                 model_size_info= [16, 3, 3, 1, 1, 1, 32, 8], 
                 is_training=True):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_size_info = model_size_info
        print(self.model_size_info)
        self.is_training = is_training

    def build_model(self):
        channels = 1
        columns = 13
        rows = int(self.input_shape / (columns * channels))
        
        input_tensor = Input(shape=(self.input_shape,))
        x = Reshape((rows, columns, channels))(input_tensor)
        x = GaussianNoise(stddev=0.45)(x)

        # CNN part
        first_filter_count = self.model_size_info[0]
        first_filter_height = self.model_size_info[1]
        first_filter_width = self.model_size_info[2]
        first_filter_stride_y = self.model_size_info[3]
        first_filter_stride_x = self.model_size_info[4]

        x = Conv2D(first_filter_count, 
                   (first_filter_height, first_filter_width), 
                   strides=(first_filter_stride_y, first_filter_stride_x), 
                   padding='valid', 
                   kernel_initializer='glorot_uniform', 
                   name='conv_1')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        if self.is_training:
            x = Dropout(0.5)(x)

        first_conv_output_height = (rows - first_filter_height + first_filter_stride_y) // first_filter_stride_y
        first_conv_output_width = (columns - first_filter_width + first_filter_stride_x) // first_filter_stride_x

        # Reshape for GRU part
        x = Reshape((first_conv_output_height, first_conv_output_width * first_filter_count))(x)

        # GRU part
        num_rnn_layers = self.model_size_info[5]
        RNN_units = self.model_size_info[6]
        

        lstm_outputs = []
        for i in range(num_rnn_layers):
            x = LSTM(RNN_units, return_sequences=True, unroll = False)(x)

            lstm_outputs.append(x)
        
        # Fully Connected part
        # add concatenate layer
        x = concatenate(lstm_outputs, axis=-1)

        first_fc_output_channels = self.model_size_info[7]
        x = Dense(first_fc_output_channels, activation='relu', kernel_initializer='glorot_uniform', name='fc1')(x)
        if self.is_training:
            x = Dropout(0.5)(x)

        # Output layer
        output_tensor = Dense(self.num_classes, activation='softmax', kernel_initializer='glorot_uniform', name='output')(x)

        
        model = Model(inputs=input_tensor, outputs=output_tensor)
        return model

if __name__ == '__main__':
    # Example usage:
    input_shape = 650
    num_classes = 3
    # model_size_info = [16, 3, 3, 1, 1, 1, 32, 8]
    model_size_info = [8, 3, 3, 1, 1, 2, 16, 8]
    # Giải thích model_size_info:
    # - 16: Số lượng bộ lọc của lớp convolution đầu tiên
    # - 3: Chiều cao của bộ lọc convolution đầu tiên
    # - 3: Chiều rộng của bộ lọc convolution đầu tiên
    # - 1: Bước dịch theo chiều dọc của bộ lọc convolution đầu tiên
    # - 1: Bước dịch theo chiều ngang của bộ lọc convolution đầu tiên
    # - 1: Số lượng lớp GRU
    # - 32: Số lượng units của lớp GRU đầu tiên
    # - 8: Số lượng units của lớp fully connected đầu tiên

    model = CRNN(input_shape, num_classes, model_size_info)
    model = model.build_model()
    model.summary()

