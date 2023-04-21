<div style="text-align: center; margin-bottom: 2.5rem">
    <h1>Model to TorchScript</h1>
    <small>Segment Anything Model (SAM)</small>
</div>

- [Introduction](#introduction)
- [Installation \& Usage](#installation--usage)
- [How to Convert your Own Models to TorchScript](#how-to-convert-your-own-models-to-torchscript)
  - [Tracing vs Scripting](#tracing-vs-scripting)
  - [Wrap the Model](#wrap-the-model)
  - [Convert the Model to TorchScript](#convert-the-model-to-torchscript)
  - [Modify/Patch the Model](#modifypatch-the-model)
    - [Tips](#tips)
  - [Save the TorchScript Model](#save-the-torchscript-model)
- [Resources](#resources)

## Introduction

This directory contains Python code to patch and save the Segment Anything Model (SAM) as TorchScript to a new file.

We used the `segment_anything` Python package provided by the [Segment Anything Model](https://github.com/facebookresearch/segment-anything) repository.

## Installation & Usage

Create a virtual environment and install the dependencies:

```bash
python3 -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt
```

Run the entry-point script:

```bash
python sam_convert.py
```

## How to Convert your Own Models to TorchScript

### Tracing vs Scripting

The PyTorch JIT api exposes two ways to convert your model to TorchScript:

- [Tracing](https://pytorch.org/docs/stable/generated/torch.jit.trace.html#torch.jit.trace)
- [Scripting](https://pytorch.org/docs/stable/generated/torch.jit.script.html#torch.jit.script)

The advantage of tracing is create a smaller and more optimized model. The disadvantage is that it only works for a subset of Python's features and does not work with conditional branches.

Scripting on the other hand, is more flexible and can handle more Python features. However, it is less optimized and can be slower than tracing.

The first step is to evaluate whether you can `trace` your model. You can read more about the limitations [here](https://pytorch.org/docs/stable/generated/torch.jit.trace.html#torch.jit.trace).

As a summary of the limitations, you cannot trace a model that:

- Has control flow (e.g. if statements) or loops
- Has data structures that are not tensors or tuples/lists/dicts of tensors
- Has function or modules that are data dependent
- Has untracked external dependencies

### Wrap the Model

If you would like to add a wrapper to the model, you can create a new model class that extends `torch.nn.Module` and add the wrapper logic in the `forward` method.

```python
class MyModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        # Add wrapper logic here
        return self.model(x)
```

You can have a look at `sam_predict_base_model.py` for an example of how to wrap the SAM model.

### Convert the Model to TorchScript

Next, you can try to convert your model to TorchScript. In this example, we used the `script` method as the SAM model has control flow in it's layers.

You will have to include example inputs to the `script` method. The TorchScript compiler will use these inputs to infer the types of the variables.

```python
model = MyModel(model)
# Set the model to evaluation mode
model.eval()

example_inputs = [
    (torch.rand(3, 256, 256),) # The tuple is the input to the forward method
] # You can add more example inputs

# Convert the model to TorchScript
scripted_model = torch.jit.script(model, example_inputs={model: example_inputs})
```

Doing for the first time will likely result in an error if you are using a model with Python specific logic. You will have to modify/patch your model to remove the unsupported features which brings us to the next step. If you do not have any errors, you can skip the next step.

### Modify/Patch the Model

In our case, the SAM model had a lot of custom logic in the `forward` method. This made it difficult to script the model. Therefore, we had to patch the model to remove the custom logic.

You will have to modify or patch a model. Modifying is the preferred approach as it is easier to maintain and keep track of changes. However, if you would like to do minimal changes without needing to maintain a fork of the model, you can patch the model.

In our case, we patched the model by using the `mock` library. The patches are located under the `patches/` directory. You can read more about the `mock` library [here](https://docs.python.org/3/library/unittest.mock.html).

Each file is named after the package's (`segment_anything`) original file name and exports a `patches` variable containing a tuple of patches.

The `patches/__init__.py` files imports these patches and applies them.

You can patch a class function the following way:

```python
import mock

example_patch = mock.patch.object(<class>, "<method name to patch>", <patched function>),

with example_patch as mock_example:
    # Do something
    pass
```

In order to avoid nested `with` statements, you can directly call the `__enter__()` method:

```python
example_patch.__enter__()

# Do something
```

#### Tips

Fixing the SAM model required a lot of trial and error. Here are some tips that might help you:

- Start converting the least amount of code, and gradually add more in. This will be easier to trace down errors.
- The error messages are not always very helpful and can sometimes be misleading. You will have to debug the code to find the error.
- Avoid any 'Pythonic' code. For example, the SAM model used lists with multiple types of data.
- Use type hints to initialize variables to None. For example, `x: torch.Tensor = None`. This will help the TorchScript compiler infer the type of the variable.

### Save the TorchScript Model

Once you have successfully converted your model to TorchScript, you can save it to a file:

```python
scripted_model.save("model.pt")
```

## Resources

- [Segment Anything Model](https://github.com/facebookresearch/segment-anything)
- [PyTorch Introduction to TorchScript](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html)
- [DJL PyTorch Engine](http://docs.djl.ai/engines/pytorch/index.html)
- [Python Mock Library](https://docs.python.org/3/library/unittest.mock.html)
