import numpy as np
import torch
import depth_pro


class DepthPro:
    def __init__(self):
        self.model, self.transform = depth_pro.create_model_and_transforms(
            device="cuda:0",
            precision=torch.half,
        )
        self.model.eval()

        # Warm up
        dummy = np.zeros((480, 854, 3), dtype=np.uint8)
        dummy = self.transform(dummy).cuda()

        with torch.inference_mode():
            for _ in range(3):
                self.model.infer(dummy, f_px=None)

        torch.cuda.synchronize()

    def predict(self, image):
        """
        Parameters
        ----------
        image : numpy.ndarray
            RGB image as H x W x 3.

        Returns
        -------
        numpy.ndarray
            Heatmap array, values normalized to [0, 1].
        """

        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a numpy.ndarray")

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Expected image shape (H, W, 3), got {image.shape}"
            )

        image = np.ascontiguousarray(image)

        tensor = self.transform(image).cuda()

        with torch.inference_mode():
            prediction = self.model.infer(
                tensor,
                f_px=None,
            )

        # Metric depth in metres
        depth = prediction["depth"].detach().cpu().numpy().squeeze()

        # Same heatmap normalization used by the Depth Pro CLI
        inverse_depth = 1.0 / depth

        max_invdepth = min(inverse_depth.max(), 1 / 0.1)
        min_invdepth = max(1 / 250, inverse_depth.min())

        heatmap = (
            inverse_depth - min_invdepth
        ) / (
            max_invdepth - min_invdepth
        )

        return heatmap.astype(np.float32)

