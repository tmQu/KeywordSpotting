from tensorflow import keras

def streaming_input_output(streaming, t, inputs, otputs, x):
  if streaming:
    otputs.append(x)
    x = keras.Input(shape=[t] + x.shape[2:])
    inputs.append(x)
  return x

def build_model(streaming=True):

  # resetting the layer name generation counter
  keras.backend.clear_session()

  inputs  = []
  outputs = []
  
  # 650 features / 50 frames = 13 features per frame

	# Input then tensor slicing
  

  x_in = keras.Input(shape=[13])
  x = x_in if streaming else keras.backend.expand_dims(x_in, -2)

  x = keras.layers.Conv2D(64, 1, use_bias=False)(x)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.ReLU()(x)
  x = keras.layers.SpatialDropout2D(x.shape[-1]/1280.0)(x)

  for i in range(4):
    x = streaming_input_output(streaming, 2, inputs, outputs, x)
    x = keras.layers.SeparableConv2D(x.shape[-1], kernel_size=[2, 1], 
                                     dilation_rate=[1 if streaming else 2**i, 1], 
                                     use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.SpatialDropout2D(x.shape[-1]/1280.0)(x)

  x = streaming_input_output(streaming, 32, inputs, outputs, x)
  x = keras.layers.AveragePooling2D([x.shape[1], 1])(x)
  x = keras.layers.Flatten()(x)

  x = keras.layers.Dense(x.shape[-1] // 2, use_bias=False)(x)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.ReLU()(x)
  x = keras.layers.Dropout(x.shape[-1]/1280.0)(x)

  x = keras.layers.Dense(12)(x)
  x = keras.layers.Softmax()(x)

  # model = keras.Model(inputs=[x_in] + inputs, outputs=[x] + outputs)
  model = keras.Model(inputs=[x_in], outputs=[x] + outputs)

  model.summary()

  return model

if __name__ == '__main__':
  model = build_model()
#   model.summary()
