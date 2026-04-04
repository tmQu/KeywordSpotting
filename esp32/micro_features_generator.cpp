/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "micro_features_generator.h"

#include <cmath>
#include <cstring>

#include "tensorflow/lite/experimental/microfrontend/lib/frontend.h"
#include "tensorflow/lite/experimental/microfrontend/lib/frontend_util.h"
#include "micro_model_settings.h"
#include "mfcc.h"
#include <Arduino.h>
#include "Wifi.h"
// #include "my_test_audio.h"
// Configure FFT to output 16 bit fixed point.
#define FIXED_POINT 16

namespace {

FrontendState g_micro_features_state;
bool g_is_first_time = true;
int16_t* test_audio;
int position = 0;

}  // namespace


static float* fe_16b_16k_mono(int16_t* samples, size_t n_samples, int *n_frames)
{
  int n_items_in_frame = 0;
  float *feat;
  feat = fe_mfcc_16k_16b_mono(samples, n_samples, n_frames, &n_items_in_frame);

  return feat;
}


float lowPassFilter(float sample, float cutoffFreq, float sampleRate) {
    static float prevSample = 0;
    static float prevOutput = 0;
    float RC = 1.0 / (cutoffFreq * 2 * 3.14159265);
    float dt = 1.0 / sampleRate;
    float alpha = dt / (RC + dt);
    float output = alpha * sample + (1 - alpha) * prevOutput;
    prevSample = sample;
    prevOutput = output;
    return output;
}

// Bộ lọc thông cao đơn giản bằng cách sử dụng công thức IIR (Infinite Impulse Response)
float highPassFilter(float sample, float cutoffFreq, float sampleRate) {
    static float prevSample = 0;
    static float prevOutput = 0;
    float RC = 1.0 / (cutoffFreq * 2 * 3.14159265);
    float dt = 1.0 / sampleRate;
    float alpha = RC / (RC + dt);
    float output = alpha * (prevOutput + sample - prevSample);
    prevSample = sample;
    prevOutput = output;
    return output;
}


float smartNoiseReducer(float sample, float noiseThreshold) {
    static float noiseLevel = 0;
    static float signalLevel = 0;

    // Tính toán mức độ tín hiệu và tiếng ồn
    if (fabs(sample) < noiseThreshold) {
        noiseLevel = 0.95 * noiseLevel + 0.05 * fabs(sample);
    } else {
        signalLevel = 0.95 * signalLevel + 0.05 * fabs(sample);
    }

    // Giảm tiếng ồn nếu mức tiếng ồn thấp hơn mức tín hiệu
    if (noiseLevel < signalLevel) {
        return sample;
    } else {
        return 0;
    }
}


TfLiteStatus InitializeMicroFeatures(tflite::ErrorReporter* error_reporter) {
  FrontendConfig config;
  config.window.size_ms = kFeatureSliceDurationMs;
  config.window.step_size_ms = kFeatureSliceStrideMs;
  config.noise_reduction.smoothing_bits = 10;
  config.filterbank.num_channels = kFeatureSliceSize;
  config.filterbank.lower_band_limit = 125.0;
  config.filterbank.upper_band_limit = 7500.0;
  config.noise_reduction.smoothing_bits = 10;
  config.noise_reduction.even_smoothing = 0.025;
  config.noise_reduction.odd_smoothing = 0.06;
  config.noise_reduction.min_signal_remaining = 0.05;
  config.pcan_gain_control.enable_pcan = 1;
  config.pcan_gain_control.strength = 0.95;
  config.pcan_gain_control.offset = 80.0;
  config.pcan_gain_control.gain_bits = 21;
  config.log_scale.enable_log = 1;
  config.log_scale.scale_shift = 6;
  if (!FrontendPopulateState(&config, &g_micro_features_state,
                             kAudioSampleFrequency)) {
    TF_LITE_REPORT_ERROR(error_reporter, "FrontendPopulateState() failed");
    return kTfLiteError;
  }
  g_is_first_time = true;
  return kTfLiteOk;
}

// This is not exposed in any header, and is only used for testing, to ensure
// that the state is correctly set up before generating results.
void SetMicroFeaturesNoiseEstimates(const uint32_t* estimate_presets) {
  for (int i = 0; i < g_micro_features_state.filterbank.num_channels; ++i) {
    g_micro_features_state.noise_reduction.estimate[i] = estimate_presets[i];
  }
}


void sendDataUDP(uint8_t *bytes, size_t count) {
    udp.beginPacket(SERVER_URL, udpServerPort);
    udp.write(bytes, count);
    udp.endPacket();

}

TfLiteStatus GenerateMicroFeatures(tflite::ErrorReporter* error_reporter,
                                  int16_t* input, int input_size,
                                   int output_size, float* output,
                                   size_t* n_out) {
  
  int16_t* frontend_input;
  int input_size_after = kFeatureSliceStrideMs * kAudioSampleFrequency / 1000;
  if (g_is_first_time) {
    frontend_input = input;
    g_is_first_time = false;
    position = 0;
  } else {
    frontend_input = input;
    // input_size_after = input_size - kFeatureSliceStrideMs * kAudioSampleFrequency / 1000;
    // if (position >= audio_len) {
    //   position = 0;
    //   delay(10000);
    // }
    // else
    // position += 16 * 100;
  }
  // test_audio = audio;
  // test_audio += position;
  // sendDataUDP((uint8_t*)frontend_input, input_size_after * 2);

  for (int i = 0; i < input_size_after; ++i) {
    int16_t sample = (int16_t) frontend_input[i];

    
    // Normalize sample
    float normalizedSample = sample / 32768.0;

    // Apply gain
    float gain = 4.5;  // Adjust gain as needed
    normalizedSample *= gain;

    // Clip if necessary
    if (normalizedSample > 1.0) {
        normalizedSample = 1.0;
    } else if (normalizedSample < -1.0) {
        normalizedSample = -1.0;
    }

    float highPassFilteredSample = highPassFilter(normalizedSample, 300, 16000);
    float bandPassFilteredSample = lowPassFilter(highPassFilteredSample, 3400, 16000);
    // float noiseReducedSample = smartNoiseReducer(bandPassFilteredSample, 0.2);

    // Chuyển đổi trở lại số nguyên 16-bit
    frontend_input[i] = (int16_t)(normalizedSample * 32768);
  }
  sendDataUDP((uint8_t*)frontend_input, input_size_after * 2);

  // Serial.write((uint8_t*)frontend_input, input_size_after * 2);
  int test = 0;
  float* feat = fe_16b_16k_mono(frontend_input, input_size_after, &test);
  if(test == 0 || test*13 != output_size) {
    ESP_LOGI("GenerateMicroFeatures", "num mfcc: %d", test*13);
  }
  for (int i = 0; i < output_size; ++i) {
    output[i] = feat[i];
  }
  
  free(feat);

  // FrontendOutput frontend_output = FrontendProcessSamples(
  //     &g_micro_features_state, frontend_input, input_size, num_samples_read);

  // for (size_t i = 0; i < frontend_output.size; ++i) {
  //   // These scaling values are derived from those used in input_data.py in the
  //   // training pipeline.
  //   // The feature pipeline outputs 16-bit signed integers in roughly a 0 to 670
  //   // range. In training, these are then arbitrarily divided by 25.6 to get
  //   // float values in the rough range of 0.0 to 26.0. This scaling is performed
  //   // for historical reasons, to match up with the output of other feature
  //   // generators.
  //   // The process is then further complicated when we quantize the model. This
  //   // means we have to scale the 0.0 to 26.0 real values to the -128 to 127
  //   // signed integer numbers.
  //   // All this means that to get matching values from our integer feature
  //   // output into the tensor input, we have to perform:
  //   // input = (((feature / 25.6) / 26.0) * 256) - 128
  //   // To simplify this and perform it in 32-bit integer math, we rearrange to:
  //   // input = (feature * 256) / (25.6 * 26.0) - 128
  //   constexpr int32_t value_scale = 256;
  //   constexpr int32_t value_div = static_cast<int32_t>((25.6f * 26.0f) + 0.5f);
  //   int32_t value =
  //       ((frontend_output.values[i] * value_scale) + (value_div / 2)) /
  //       value_div;
  //   value -= 128;
  //   if (value < -128) {
  //     value = -128;
  //   }
  //   if (value > 127) {
  //     value = 127;
  //   }
  //   // calculate mfcc
  //   output[i] = value;
  // }

  return kTfLiteOk;
}



