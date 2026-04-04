import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Reshape



#default CNN model on Edge impulse
class CNN:
    def __init__(self, input_length, classes):
        self.input_length = input_length
        self.classes = classes
        self.model = None

    #copy from Edge impulse
    def build_model(self):
        # model architecture
        self.model = Sequential()
        # Data augmentation, which can be configured in visual mode
        self.model.add(tf.keras.layers.GaussianNoise(stddev=0.45))
        channels = 1
        columns = 13
        rows = int(self.input_length / (columns * channels))
        self.model.add(Reshape((rows, columns, channels), input_shape=(self.input_length, )))
        self.model.add(Conv2D(8, kernel_size=3, kernel_constraint=tf.keras.constraints.MaxNorm(1), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
        self.model.add(Dropout(0.5))
        self.model.add(Conv2D(16, kernel_size=3, kernel_constraint=tf.keras.constraints.MaxNorm(1), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
        self.model.add(Dropout(0.5))
        self.model.add(Flatten())
        self.model.add(Dense(20, activation='relu',
            activity_regularizer=tf.keras.regularizers.l1(0.00001)))
        self.model.add(Dense(self.classes, name='y_pred', activation='softmax'))
        # self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 
        self.model.build((None, self.input_length))
        return self.model



# model = CNN(650, 3)
# model = model.build_model()
# model.summary()