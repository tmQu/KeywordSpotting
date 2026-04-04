#include <TensorFlowLite_ESP32.h>
/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// #include "main_functions.h"

#include "audio_provider.h"
#include "command_responder.h"
#include "feature_provider.h"
#include "micro_model_settings.h"
#include "model.h"
#include "recognize_commands.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "mfcc.h"

#include "Wifi.h"

WiFiUDP udp;


#define NUM_TENSOR_INPUT 5
// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
FeatureProvider* feature_provider = nullptr;
RecognizeCommands* recognizer = nullptr;
int32_t previous_time = 0;
int32_t count_run = 0;

// Create an area of memory to use for input, output, and intermediate arrays.
// The size of this will depend on the model you're using, and may need to be
// determined by experimentation.
constexpr int kTensorArenaSize = 20 * 1024;
uint8_t *tensor_arena;
float feature_buffer[kFeatureElementCount];

int input_shape[NUM_TENSOR_INPUT] = {65, 4*65, 2*62*16, 6*61*16, 16};
// int input_shape[NUM_TENSOR_INPUT] = {130, 4*130, 2*127*16, 1*126*4, 4};
float *inputs_values[NUM_TENSOR_INPUT];

TfLiteTensor *input[NUM_TENSOR_INPUT];
TfLiteTensor* output[NUM_TENSOR_INPUT];


unsigned long count = 0;  
float probability[3] = {0.0f, 0.0f, 0.0f};
}  // namespace
#define TEST false


void wifi_task(void *pvParameters)
{
  
  WiFi.mode(WIFI_STA);
  WiFi.begin(SSID, PASSWORD);
  while (WiFi.waitForConnectResult() != WL_CONNECTED) {
      Serial.print(".");
      vTaskDelay(1000/portTICK_PERIOD_MS);
  }
  udp.begin(3333);

  vTaskDelete(NULL);
}
// The name of this function is important for Arduino compatibility.
void setup() {
  Serial.begin(115200);
  xTaskCreate(wifi_task, "wifi_task", 4096, NULL, 1, NULL);
  // WiFi.begin(SSID, PASSWORD);
  // while (WiFi.waitForConnectResult() != WL_CONNECTED) {
  //     Serial.print(".");
  //     delay(1000);
  // }
  // // WiFi.deinit();
  delay(3000);  
  // xTaskCreate(test_task, "test_task", 4096, NULL, 1, NULL);
  Serial.println("Connected to WiFi");
  // udp.begin(por)

  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }
  if (tensor_arena == NULL) {
    tensor_arena = (uint8_t *) heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  }


  if (tensor_arena == NULL) {
    Serial.println("Couldn't allocate memory of bytes");
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  //
  // tflite::AllOpsResolver resolver;
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroMutableOpResolver<14> micro_op_resolver(error_reporter);
    micro_op_resolver.AddAveragePool2D(); \
    micro_op_resolver.AddConcatenation(); \
    micro_op_resolver.AddConv2D(); \
    micro_op_resolver.AddDepthwiseConv2D(); \
    micro_op_resolver.AddDequantize(); \
    micro_op_resolver.AddFullyConnected(); \
    micro_op_resolver.AddReshape(); \
    micro_op_resolver.AddSoftmax(); \
    micro_op_resolver.AddStridedSlice(); \
    micro_op_resolver.AddQuantize(); \
    micro_op_resolver.AddCallOnce(); \
    micro_op_resolver.AddVarHandle(); \
    micro_op_resolver.AddReadVariable(); \
    micro_op_resolver.AddAssignVariable();

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Get information about the memory area to use for the model's input.
  // model_input = interpreter->input(0);
  // if ((model_input->dims->size != 2) || (model_input->dims->data[0] != 1) ||
  //     (model_input->dims->data[1] !=
  //      (kFeatureSliceCount * kFeatureSliceSize)) ||
  //     (model_input->type != kTfLiteInt8)) {
  //   TF_LITE_REPORT_ERROR(error_reporter,
  //                        "Bad input tensor parameters in model");
  //   return;
  // }
  // // model_input_buffer = model_input->data.int8;
  // input[0] = model_input;

  for (int i = 0; i < NUM_TENSOR_INPUT; i++)
  {
    input[i] = interpreter->input(i);
    output[i] = interpreter->output(i);
    inputs_values[i] = (float *)heap_caps_malloc(input_shape[i] * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    for (int j = 0; j < input_shape[i]; j++)
    {
      inputs_values[i][j] = 0;
    }
    if (inputs_values[i] == NULL)
    {
      TF_LITE_REPORT_ERROR(error_reporter, "Failed to allocate memory for input tensor");
      return;
    }
  }

  // Prepare to access the audio spectrograms from a microphone or other source
  // that will provide the inputs to the neural network.
  // NOLINTNEXTLINE(runtime-global-variables)
  static FeatureProvider static_feature_provider(kFeatureElementCount,
                                                 feature_buffer);
  feature_provider = &static_feature_provider;

  static RecognizeCommands static_recognizer(error_reporter);
  recognizer = &static_recognizer;

  previous_time = 0;

  // print all the input, output tensor

}

// The name of this function is important for Arduino compatibility.
void loop() {
  // Fetch the spectrogram for the current time.
  // for (int i = 0; i < NUM_TENSOR_INPUT; i++)
  // {
  //   if (input[i]->type == kTfLiteInt8 || input[i]->type == kTfLiteUInt8)
  //   {
  //     Serial.printf("Input tensor %d: %s, %d\n", i, input[i]->name, input[i]->bytes / sizeof(int8_t));
  //   }
  //   else
  //   {
  //     Serial.printf("Input tensor %d: %s, %d\n", i, input[i]->name, input[i]->bytes / sizeof(float));
  //   }


  // }
  unsigned long start_dsp_time = millis();
  // Serial.printf("test_time %d\n", start_dsp_time - test_time);

  const int32_t current_time = LatestAudioTimestamp();
  int how_many_new_slices = 0;
  // Serial.printf("test_time %d, current time %d, prev time %d, millise %d\n", start_dsp_time - test_time, current_time, previous_time, millis()); 

  int feature_status = feature_provider->PopulateFeatureData(
      error_reporter, previous_time, current_time, &how_many_new_slices);

  
  // if (feature_status != kTfLiteOk) {
  //   TF_LITE_REPORT_ERROR(error_reporter, "Feature generation failed");
  //   return;
  // }
  if (how_many_new_slices == 0) {
    return;
  }
  // Serial.printf("test_time %d, current time %d, prev time %d, current %d, millise %d\n", start_dsp_time - test_time, current_time, previous_time, LatestAudioTimestamp(),millis()); 

  // unsigned long end_dsp_time = millis();
  // // Serial.printf("current_time: %d\n", current_time - previous_time);
  // // Serial.print("DSP time (ms): ");
  // // Serial.println(end_dsp_time - start_dsp_time);

  previous_time += (feature_status) * 1000 / kAudioSampleFrequency;
  // Serial.printf("current_time: %d, prev time %d, bytes %d\n", current_time, previous_time, feature_status);

  // If no new audio samples have been received since last time, don't bother
  // running the network model.

  
  // Copy feature buffer to input tensor
  for (int i = 0; i < kFeatureElementCount; i++) {

    if (input[0]->type == kTfLiteInt8 || input[0]->type == kTfLiteUInt8)
    {
      if (kFeatureElementCount != input[0]->bytes / sizeof(int8_t))
      {
        TF_LITE_REPORT_ERROR(error_reporter, "Input tensor int size mismatch %d", input[0]->bytes / sizeof(int8_t));
      }
      input[0]->data.int8[i] = static_cast<int8_t>(round((feature_buffer[i] / input[0]->params.scale) + input[0]->params.zero_point));
      if (TEST && i < 10)
      {
        Serial.print("Input tensor ");
        Serial.print(static_cast<int8_t>(round((feature_buffer[i] / input[0]->params.scale) + input[0]->params.zero_point)));
        Serial.print(" ");
      }
    }
    else
    {
      if (kFeatureElementCount != input[0]->bytes / sizeof(float))
      {
        TF_LITE_REPORT_ERROR(error_reporter, "Input tensor int size mismatch %d", input[0]->bytes / sizeof(int8_t));
      }
      input[0]->data.f[i] = feature_buffer[i];
    }
  }
  if (TEST)
  {
    Serial.println();
  }
  unsigned long start_time = millis();
  for (int i = 1; i < NUM_TENSOR_INPUT; i++)
  {
    // Check the input size

    // memcpy(input[i]->data.f, inputs_values[i], input_shape[i] * sizeof(float));
    if (input[i]->type == kTfLiteInt8 || input[i]->type == kTfLiteUInt8)
    {
      for (int j = 0; j < input_shape[i]; j++)
      {
        input[i]->data.int8[j] = static_cast<int8_t>(round(inputs_values[i][j] / input[i]->params.scale + input[i]->params.zero_point));
      }
          if (input_shape[i] != input[i]->bytes / sizeof(int8_t))
      {
        TF_LITE_REPORT_ERROR(error_reporter, "%d Input tensor int size mismatch %d", i, input[i]->bytes / sizeof(int8_t));
      }
    }
    else
    {
      if(input_shape[i] != input[i]->bytes / sizeof(float))
      {
        TF_LITE_REPORT_ERROR(error_reporter, "%d Input tensor size mismatch %d", i, input[i]->bytes / sizeof(float)); 
      }

      for (int j = 0; j < input_shape[i]; j++)
      {
        input[i]->data.f[j] = inputs_values[i][j];
      }
    }

  }

  // Run the model on the spectrogram input and make sure it succeeds.
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
    return;
  }


  // sAVE THE OUTPUT OF THE MODEL
  for (int i = 1; i < NUM_TENSOR_INPUT; i++)
  {
    
    // memcpy(inputs_values[i], output[i]->, input_shape[i] * sizeof(float));
    for (int j = 0; j < input_shape[i]; j++)
    {
      // inputs_values[i][j] = output[i]->params.scale * (output[i]->data.int8[j] - output[i]->params.zero_point);
      inputs_values[i][j] = (output[i]->data.int8[j] * 1.0f - output[i]->params.zero_point) * output[i]->params.scale;

    }

  }
  // Measure the end time
  unsigned long end_time = millis();

  // Calculate the elapsed time
  unsigned long elapsed_time = end_time - start_time;

  // Log the elapsed time to the serial monitor
  // Serial.printf("Inference time (ms): %d, dsp time %d\n", elapsed_time, end_dsp_time - start_dsp_time);
  // Serial.println(elapsed_time);


  // Obtain a pointer to the output tensor
  // TfLiteTensor* output = interpreter->output(0);
  // Serial.println("testing");
  int max_index = 0;
  int max_value = -1;
  // Serial.println("-------------------------------------------------------------------------");
  for (int i = 0; i < kCategoryCount; i++)
  {
    float portion = (output[0]->data.int8[i] * 1.0f - output[0]->params.zero_point)*output[0]->params.scale;
    probability[i] += portion;
    if(i == 0 && portion > 0.8) { // my keyword, Noise, unknown
    Serial.print(kCategoryLabels[i]);
    Serial.printf("%f ", portion);
    Serial.println(" ");
    }
    // Serial.print(kCategoryLabels[i]);
    // Serial.print(": ");
    // Serial.println(portion);

    if (portion > max_value)
    {
      max_value = portion;
      max_index = i;
    }

  }
  count++;




  // Serial.printf("Time taken: %d ms, dsp time %d, millis %d\n", millis() - start_dsp_time, elapsed_time, millis());

  
  // Determine whether a command was recognized based on the output of inference
  // const char* found_command = nullptr;
  // uint8_t score = 0;
  // bool is_new_command = false;
  // TfLiteStatus process_status = recognizer->ProcessLatestResults(
  //     output, current_time, &found_command, &score, &is_new_command);
  // if (process_status != kTfLiteOk) {
  //   TF_LITE_REPORT_ERROR(error_reporter,
  //                        "RecognizeCommands::ProcessLatestResults() failed");
  //   return;
  // }
  // // Do something based on the recognized command. The default implementation
  // // just prints to the error console, but you should replace this with your
  // // own function for a real application.
  // RespondToCommand(error_reporter, current_time, found_command, score,
  //                  is_new_command);
}
