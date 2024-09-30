import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np

X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')

X_train = np.float32(X_train)
# y_train = np.int32(y_train)
# Parameters
segment_length = 26
num_segments = 650 // segment_length

# Reshape X_train
# Reshape each row of X_train into 50 segments of length 13
X_train_reshaped = X_train.reshape(-1, num_segments, segment_length)

# Reshape y_train
# Replicate each label 50 times
y_train_replicated = np.repeat(y_train, num_segments)

# Reshape X_train_reshaped to the desired shape
X_train_final = X_train_reshaped.reshape(-1, segment_length)

print("X_train_final shape:", X_train_final.shape)  # Expected shape: (data_length * 50, 13)
print("y_train_replicated shape:", y_train_replicated.shape)  # Expected shape: (data_length * 50,)



X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

train_dataset = tf.data.Dataset.from_tensor_slices((X_train_final, y_train_replicated))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

def representative_data_gen():
  for input_value, _ in train_dataset.take(100):

    yield [input_value]

model = tf.keras.models.load_model('model_stream_2')
model.summary()

# find the dynamic tensor
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.InputLayer):
        print(layer.output)
        break
    


converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model('model_stream_2')
converter.experimental_new_converter = True
converter._enable_tflite_resource_variables = True
converter.representative_dataset = representative_data_gen
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()
open("model_stream.tflite", 'wb').write(tflite_model)
print('Model size is %f MBs.' % (len(tflite_model) / 1024 / 1024.0))

print('hello')


# import tensorflow as tf
# import tensorflow_model_optimization as tfmot
# import numpy as np

# # Bước 1: Tải mô hình từ thư mục SavedModel
# model_path = 'model_stream'
# your_model = tf.keras.models.load_model(model_path)


# def annotate(layer):
#   if layer._name.startswith('MyReshape'):
#     return layer
#   # quantize everything else
#   return tfmot.quantization.keras.quantize_annotate_layer(layer)
# # Bước 2: Áp dụng quantize-aware training 
# q_aware_model = tfmot.quantization.keras.quantize_model(your_model)

# # Bước 3: Biên dịch lại mô hình
# q_aware_model.compile(optimizer='adam',
#                       loss='sparse_categorical_crossentropy',
#                       metrics=['accuracy'])

# # Giả sử bạn có train_dataset và validation_dataset
# # Bước 4: Huấn luyện lại mô hình

# X_train = np.load('X_train.npy')
# y_train = np.load('y_train.npy')

# X_test = np.load('X_test.npy')
# y_test = np.load('y_test.npy')

# train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
# validation_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

# q_aware_model.fit(train_dataset, epochs=5, validation_data=validation_dataset)

# # Bước 5: Chuyển đổi mô hình đã huấn luyện với quantize-aware training
# converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_ops = [tf.lite.OpsSet.SELECT_TF_OPS]
# tflite_qat_model = converter.convert()

# # Lưu mô hình đã chuyển đổi
# with open("model_qat.tflite", 'wb') as f:
#     f.write(tflite_qat_model)

# print('Model size is %f MBs.' % (len(tflite_qat_model) / 1024 / 1024.0))
