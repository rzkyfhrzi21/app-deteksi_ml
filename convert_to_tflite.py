import tensorflow as tf

model = tf.keras.models.load_model("model.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

with open("model_fp16.tflite", "wb") as f:
    f.write(tflite_model)

print("model_fp16.tflite selesai")
