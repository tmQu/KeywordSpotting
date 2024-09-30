# Keyword Spotting Project

This project is a **Keyword Spotting** system based on [Google's Keyword Spotting Streaming]("https://github.com/google-research/google-research/blob/master/kws_streaming/README.md), [(paper)]("https://arxiv.org/abs/2005.06720"). The goal of the project is to recognize predefined keywords in real-time using audio streaming. The system is optimized for deployment on edge devices such as microcontrollers (ESP32).

## Features

- **Real-time Keyword Detection**: Continuously monitors audio input for specific keywords.
- **Streaming Model**: Uses Google's keyword spotting streaming model for low-latency, efficient processing.
- **Edge-Optimized**: Model is designed for deployment on low-resource devices (ESP32).
- **MFCC Features**: Utilizes Mel-frequency cepstral coefficients (MFCC) for feature extraction.

## Prerequisites

To run or deploy the project, you will need:

- **Python 3.10**
- **TensorFlow/Keras** for model training and inference
- **Librosa** for audio processing and MFCC extraction


## Installation

1. Clone the repository:
2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training the Model

1. Preprocess your audio data into MFCC features (preprocessing_data.ipynb)
2. Train the model using the streaming method (stream_model.ipynb)
3. Save the trained model in TensorFlow Lite format for deployment.

### Deploying to ESP32

1. Convert TensorFlow Lite format to model.cc using the following command:

    ```bash
        xxd -i model_stream_external.tflite > model.cc
    ```

2. Flash the model onto the ESP32.

## References

- [Google's Keyword Spotting Streaming]("https://github.com/google-research/google-research/blob/master/kws_streaming/README.md)

