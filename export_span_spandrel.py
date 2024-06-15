from spandrel import ModelLoader
import torch
import sys
import os
# load a model from disk
try:
    model_str = sys.argv[1]
except:
    model_str = input("Paste model path/name here: ")

conversion_path = "converted_models/"

model = ModelLoader().load_from_file(model_str)



if not os.path.exists(conversion_path): os.mkdir(conversion_path)

# get model attributes
scale = model.scale


model = model.model

state_dict = model.state_dict()

model.eval()
model.load_state_dict(state_dict, strict=True)

print("Exporting...")

with torch.inference_mode():
    mod = torch.jit.trace( model,
        torch.rand(1, 3, 32, 32))
    mod.save(os.path.join(f'{conversion_path}',f'{scale}x_{model_str}.pt'))
    torch.onnx.export(
        model,
        torch.rand(1, 3, 32, 32),
        os.path.join(f'{conversion_path}',f'{scale}x_{model_str}.onnx'),
        verbose=False,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        #dynamic_axes={
        #        "input": {0: "batch_size", 2: "width", 3: "height"},
        #        "output": {0: "batch_size", 2: "width", 3: "height"},
        #    }
    )
