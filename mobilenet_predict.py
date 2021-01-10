# MIT License
#
# Copyright (c) 2021 Fagner Cunha
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

r"""Tool to predict image label using TensorFlow Lite.

Example Usage:
python3 mobilenet_predict.py \
    --model=/tmp/mobilenetv2_cats_and_dogs_keras.tflite \
    --images_patern=/tmp/samples/* \
    --label_map=cats,dogs
"""
import glob
import time

from absl import app
from absl import flags

from PIL import Image
import tflite_runtime.interpreter as tfl
import numpy as np

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'model', default=None,
    help=('File path of .tflite file.'))

flags.DEFINE_string(
    'images_patern', default=None,
    help=('A file pattern for images to be used during evaluation.'))

flags.DEFINE_list(
    'label_map', default=None,
    help=('List of classes to map predictions.'))

flags.mark_flag_as_required('model')
flags.mark_flag_as_required('images_patern')

def load_and_preprocess_image(image_path, height, width):
  image = Image.open(image_path).convert('RGB')
  image = image.resize((height, width), Image.ANTIALIAS)
  image = np.asarray(image, dtype=np.float32)
  image = image/255
  image = np.expand_dims(image, axis=0)

  return image

def load_model(model_path):
  interpreter = tfl.Interpreter(model_path)
  interpreter.allocate_tensors()

  _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

  return interpreter, input_height, input_width

def classify_image(interpreter, image):
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  interpreter.set_tensor(input_details[0]['index'], image)
  interpreter.invoke()
  predictions = interpreter.get_tensor(output_details[0]['index'])

  return predictions[0]

def predict(model_path, image_list):
  image_count = 0
  total_elapsed_time = 0
  interpreter, input_height, input_width = load_model(model_path)

  for image_path in image_list:
    start_time = time.time()

    image = load_and_preprocess_image(image_path, input_height, input_width)
    prediction = classify_image(interpreter, image)

    elapsed_ms = (time.time() - start_time) * 1000
    total_elapsed_time += elapsed_ms
    image_count += 1

    label_pred = np.argmax(prediction)
    confidence = prediction[label_pred]
    if FLAGS.label_map:
      label_pred = FLAGS.label_map[label_pred]
    print("Image: %s, predicted class: %s, confidence: %.2f%%" % 
                (image_path, str(label_pred), 100*confidence))

  return total_elapsed_time/image_count

def main(_):
  image_list = glob.glob(FLAGS.images_patern)
  avg_elapsed_time = predict(FLAGS.model, image_list)
  print("Averaged elapsed time: %fms" % (avg_elapsed_time))

if __name__ == '__main__':
  app.run(main)
