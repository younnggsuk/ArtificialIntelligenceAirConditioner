import os

hand_sign_model_path = ""

model = tf.keras.models.load_model(os.path.join(hand_sign_model_path, 'hand_sign.h5'))

# Keras의 모델(*.h5)을 Tensorflow의 Saved Model 형태로 바꿔서 다시 저장
model.save(os.path.join(root_path, 'hand_sign_saved_model'))

# TF-TRT 모델로 변환

from tensorflow.python.compiler.tensorrt import trt_convert as trt

conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode=trt.TrtPrecisionMode.FP16,
                                                               max_workspace_size_bytes=1<<25,
                                                               max_batch_size=1,
                                                               minimum_segment_size=50)

converter = trt.TrtGraphConverterV2(input_saved_model_dir=os.path.join(root_path, 'hand_sign_saved_model'),
                                    conversion_params=conversion_params)

converter.convert()
converter.save(output_saved_model_dir=os.path.join(root_path, 'hand_sign_tf_trt_FP16'))