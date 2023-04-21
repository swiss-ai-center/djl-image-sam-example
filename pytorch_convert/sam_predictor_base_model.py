import torch
from torch import nn
from segment_anything.modeling.sam import Sam
from patches.predictor import SamPredictorMock as SamPredictor


class SamPredictorBaseModel(nn.Module):
    """
    A wrapper around the SAM model that allows it to be used as a TorchScript model.
    """

    def __init__(self, model: Sam) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        image: torch.Tensor,
    ):
        """
        Predicts as mask end-to-end from provided image and the center of the image.

        Args:
            image (torch.Tensor): Input image, in shape 3xHxW.

        Returns:
            (dict): A dictionary with the following keys.
                'masks': (torch.Tensor) Binary mask predictions,
                with shape CxHxW
                'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape C.
                'low_res_logits': (torch.Tensor) The model's predictions
                of mask quality, in shape CxHxW, where H=W=256.
        """
        center = torch.IntTensor([image.shape[-2] // 2, image.shape[-1] // 2])

        # Unsqueeze center to add BxN dimension, where B=N=1
        center = center[None, None, :]

        point_labels = torch.IntTensor([0])
        point_labels = point_labels[None, :]

        image = image[None, :]

        input = (
            image,
            [
                {
                    "original_size": torch.IntTensor([1024, 1024]),
                    "point_coords": center,
                    "point_labels": point_labels,
                }
            ],
            True,
        )
        output = self.model(*input)
        batch_output = output[0]

        # Remove batch dimensions from model ouputs
        return {
            "masks": batch_output["masks"].squeeze(0),
            "iou_predictions": batch_output["iou_predictions"].squeeze(0),
            "low_res_logits": batch_output["low_res_logits"].squeeze(0),
        }
