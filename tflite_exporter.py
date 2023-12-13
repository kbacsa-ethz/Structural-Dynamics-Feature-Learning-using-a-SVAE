import os
import numpy as np
import tensorflow as tf


def export_model(model, steps, input_size, save_path, calibrate_dataset):

    run_model = tf.function(lambda x: model(x))
    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec([1, steps, input_size], model.inputs[0].dtype))

    # model directory.
    model.save(save_path, save_format="tf", signatures=concrete_func)
    converter = tf.lite.TFLiteConverter.from_saved_model(save_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    dataset_size = tf.data.experimental.cardinality(calibrate_dataset)

    def data_generator():
        for sample in calibrate_dataset.take(dataset_size):
            data, label = sample
            yield [data]

    converter.representative_dataset = data_generator
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8  # or tf.uint8
    converter.inference_output_type = tf.int8  # or tf.uint8
    tflite_model = converter.convert()

    # Save the model.
    with open(os.path.join(save_path, 'model.tflite'), 'wb') as f:
        f.write(tflite_model)

    # Run the model with TensorFlow Lite
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_scale, input_zero_point = input_details[0]['quantization']
    output_scale, output_zero_point = output_details[0]['quantization']

    for x_test in calibrate_dataset.take(dataset_size):
        x_data, x_label = x_test
        expected = model(x_data)
        x_test_quant = np.int8(x_data / input_scale + input_zero_point)
        interpreter.set_tensor(input_details[0]["index"], x_test_quant)
        interpreter.invoke()
        result_quant = interpreter.get_tensor(output_details[0]["index"])
        result = ((result_quant - output_zero_point) * output_scale).astype(float)

        # Assert if the result of TFLite model is consistent with the TF model.
        try:
            np.testing.assert_almost_equal(expected, result, decimal=2)
        except Exception as e:
            print(e)

        # Please note: TfLite fused Lstm kernel is stateful, so we need to reset
        # the states.
        # Clean up internal states.
        interpreter.reset_all_variables()
