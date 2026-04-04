import tensorflow as tf
from tensorflow.keras.layers import AveragePooling2D, Reshape, Conv1D, BatchNormalization, ReLU, Add, AveragePooling1D, Flatten, Dropout, Dense, Input, Conv2D
from tensorflow.keras.models import Model

class TC_Resnet:
    def residual_block_s1(input_tensor, c, k):
        filter = int(c * k)
        x = Conv2D(filter, 9, strides=1, use_bias=False, padding='same')(input_tensor)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(filter, 9, strides=1, use_bias=False, padding='same')(x)
        x = BatchNormalization()(x)
        x = Add()([x, input_tensor])
        x = ReLU()(x)
        return x


    def residual_block_s2(self, input_tensor, c, k):
        filter = int(c * k)
        x1 = Conv2D(filter, (9, 1), 
                    strides=2,
                    use_bias=False, 
                    padding='same')(input_tensor)
        x1 = BatchNormalization()(x1)
        x1 = ReLU()(x1)
        x1 = Conv2D(filter, (9, 1), 
                    strides=1,
                    use_bias=False, 
                    padding='same')(x1)
        x1 = BatchNormalization()(x1)

        x2 = Conv2D(filter, 1, 
                    strides=2,
                    use_bias=False, 
                    padding='same')(input_tensor)
        x2 = BatchNormalization()(x2)
        x2 = ReLU()(x2)

        x = Add()([x1, x2])
        x = ReLU()(x)
        return x
    
    def build_tc_resnet_8(self, input_shape, num_classes, k=1):
        channels = 1
        columns = 13
        rows = int(input_shape / (columns * channels))
        input_layer = Input(input_shape)
        x = Reshape((rows, columns, channels))(input_layer)
        x = Conv2D(int(16 * k), (3, 1), strides=1, use_bias=False,
                padding='same')(x)
        x = self.residual_block_s2(x, 24, k)
        x = self.residual_block_s2(x, 32, k)
        x = self.residual_block_s2(x, 48, k)
        x = AveragePooling2D(pool_size=(3, 1))(x)
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        output_layer = Dense(num_classes, activation='softmax')(x)
        return Model(inputs=input_layer, outputs=output_layer)

    def build_smaller_tc_resnet(self, input_shape, num_classes, k=1):
        channels = 1
        columns = 13
        rows = int(input_shape / (columns * channels))
        input_layer = Input(input_shape)
        x = Reshape((rows, columns, channels))(input_layer)
        x = Conv2D(int(8 * k), (3, 1), strides=1, use_bias=False,
                padding='same')(x)
        x = self.residual_block_s2(x, 8, k)
        x = self.residual_block_s2(x, 16, k)
        x = self.residual_block_s2(x, 24, k)
        x = AveragePooling2D(pool_size=(3,1))(x)
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        output_layer = Dense(num_classes, activation='softmax')(x)
        return Model(inputs=input_layer, outputs=output_layer)
    

if __name__ == '__main__':
    model = TC_Resnet()
    model = model.build_smaller_tc_resnet(650, 3, 0.5)
    model.summary()

