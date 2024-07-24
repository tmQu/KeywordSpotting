import tensorflow as tf
import librosa
from mfcc import generate_features
import numpy as np

model_ds_cnn = tf.keras.models.load_model('E:/KeywordSpotting/self/gen_model/get_model/train_model/model_ds_cnn.keras')

# MinhQuang Noise Unknown

def extract_mfcc(audio_path):
    audio, sr = librosa.load(audio_path, sr=16000, duration=1)

    # Số lượng coefficient MFCC
    n_mfcc = 13

    # Độ dài của frame và bước nhảy của frame (chuyển đổi sang số mẫu)
    frame_length = 0.02  # 20 ms
    frame_stride = 0.02  # 20 ms

    
    # Số lượng bộ lọc Mel (tương đương với filter_number)
    filter_number = 32

    # Độ dài của cửa sổ Fourier (tương đương với fft_length)
    n_fft = 256
    # Calculate MFCC features
    # Trích xuất MFCC features
    mfccs = generate_features(implementation_version=4,
                                win_size=101,
                                pre_shift=1,
                                draw_graphs=False,
                                raw_data=audio,
                                axes=['accY'],
                                sampling_freq=sr,
                                frame_length=frame_length,
                                frame_stride=frame_stride,
                                num_filters=filter_number,
                                fft_length=n_fft,
                                low_frequency=0,
                                high_frequency=0,
                                pre_cof=0.98,
                                num_cepstral=n_mfcc
                                )

    # Normalize MFCCs (optional)
    # mfccs = librosa.util.normalize(mfccs, axis=1, norm=1)
    mfccs = np.array(mfccs['features'])
    return mfccs

# load audio
audio_path = 'get_model//raw_data/MinhQuang//20127133_Minh Quang_rec1.wav'
audio, sr = librosa.load(audio_path, sr=16000)



# extract MFCC
data = extract_mfcc(audio_path)
print(data.shape)

# predict
data = np.expand_dims(data, axis=0)

# xs = 1

model_ds_cnn.predict(data)

# print result
print(model_ds_cnn.predict(data))