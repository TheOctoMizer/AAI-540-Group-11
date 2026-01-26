from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.quant_utils import optimize_model

def quantize_onnx(fp32_path, int8_path):
    # Generate INT8 quantized model
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
    )
