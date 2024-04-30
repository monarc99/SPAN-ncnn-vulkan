from spandrel import ModelLoader
import torch

# load a model from disk
model_str = "2x_ModernSpanimationV1.pth"
model = ModelLoader().load_from_file(model_str)


model = model.model
state_dict = model.state_dict()

model.eval()
model.load_state_dict(state_dict, strict=True)

print("Exporting...")

with torch.inference_mode():
    mod = torch.jit.trace( model,
        torch.rand(1, 3, 256, 256))
    mod.save('ModernSpanimationV1.pt')
    torch.onnx.export(
        model,
        torch.rand(1, 3, 256, 256),
        "ModernSpanimationV1.onnx",
        verbose=False,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        #dynamic_axes={
        #        "input": {0: "batch_size", 2: "width", 3: "height"},
        #        "output": {0: "batch_size", 2: "width", 3: "height"},
        #    }
    )
