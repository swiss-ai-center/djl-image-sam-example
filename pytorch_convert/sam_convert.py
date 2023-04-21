import os
import torch
from segment_anything import sam_model_registry
from patches import apply_patches
from sam_predictor_base_model import SamPredictorBaseModel


def main():
    """
    Convert SAM to TorchScript and save it.

    See: http://docs.djl.ai/docs/pytorch/how_to_convert_your_model_to_torchscript.html
    """
    if not os.path.exists("sam_vit_b_01ec64.pth"):
        os.system(
            "wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        )

    apply_patches()

    # An instance of the model.
    base_model = sam_model_registry["vit_b"](checkpoint="./sam_vit_b_01ec64.pth")

    # An example input you would normally provide to your model's forward() method.
    B = 1
    N = 1
    H = 1024
    W = 1024
    example_inputs = [(torch.randint(0, 255, size=(3, H, W)),)]

    model = SamPredictorBaseModel(model=base_model)
    model.eval()

    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    # This is commented as it does work work with the SAM model.
    # traced_script_module = torch.jit.trace(naive_model, example_inputs)

    scripted_model = torch.jit.script(model, example_inputs={model: example_inputs})
    # This is also commented as it does work work with the SAM model.
    # script_module = torch.jit.optimize_for_inference(script_module)

    # Preview the TorchScript model
    print(scripted_model(*example_inputs[0]))

    # Save the TorchScript model
    scripted_model.save("../src/resources/pytorch_models/sam_vit_b/sam_vit_b.pt")


if __name__ == "__main__":
    main()
