import torch

def export_onnx(model, input_dim, output_path):
    model.eval()

    dummy_input = torch.randn(1, input_dim)
    
    # Define dynamic axes for batch dimension
    dynamic_axes = {
        'input': {0: 'batch_size'},
        'reconstruction': {0: 'batch_size'}
    }

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["reconstruction"],
        dynamic_axes=dynamic_axes,
        dynamo=False,  # Disable dynamo to avoid shape inference issues
    )