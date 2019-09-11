import torch
import torch.nn as nn
from torchsummary import summary
import torch.onnx
import onnx
import onnxruntime

from torch.autograd import Variable

import model
import dataset

import argparse
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m',
        dest='model_path',
        required=True,
        help="Path to the trained model"
    )
    
    args = parser.parse_args()

    # device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load model
    model = model.PilotNet(input_shape=(66, 200))
    model.train(False)
      
    if os.path.exists(args.model_path) == False:
        print("Error: model not found!")
        exit(1)
    else:
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    # input to the model
    x = Variable(torch.randn(1, 3, 66, 200, device=device))
    #torch_out = model(x)

    # save to train -> .pth
    # save to deloy -> .onnx
    output_model_path = args.model_path.split('.')[0] + ".onnx"


    # export the model
    torch.onnx.export(model,
                      x,
                      output_model_path,
                      #export_params=True,
                      #opset_version=7,
                      #do_constant_folding=True,
                      #input_names = ['input'], 
                      #output_names = ['output'],
                      operator_export_type=torch._C._onnx.OperatorExportTypes.ONNX,
                      verbose=True
    )

    # check model
    onnx_model = onnx.load(output_model_path)
    onnx.checker.check_model(onnx_model)

    # test runtime
    ort_session = onnxruntime.InferenceSession(output_model_path)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    rt_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")