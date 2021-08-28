import torch
import numpy as np

class Normalize(torch.nn.Module):
    """Normalization layer as described in ParseNet paper.

    # Arguments
        scale: Default feature scale.

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)

    # Output shape
        Same as input

    # References
        http://cs.unc.edu/~wliu/papers/parsenet.pdf

    """
    def __init__(self, scale, input_shape, name, **kwargs):
        super(Normalize, self).__init__(**kwargs)
        self.scale = scale
        self.name = '{}_gamma'.format(name)

        shape = (1, input_shape[1],) + tuple([1] * (len(input_shape) - 1))
        init_gamma = torch.tensor(self.scale * np.ones(shape, dtype=np.float32))
        self.register_parameter(self.name, torch.nn.Parameter(init_gamma))

    def call(self, x, mask=None):
        output = x / torch.norm(x, dim=1, keepdim=True)

        gamma = getattr(self, self.name)

        output *= gamma
        return output

class PriorBox(torch.nn.Module):
    """Generate the prior boxes of designated sizes and aspect ratios.

    # Arguments
        img_size: Size of the input image as tuple (w, h).
        min_size: Minimum box size in pixels.
        max_size: Maximum box size in pixels.
        aspect_ratios: List of aspect ratios of boxes.
        flip: Whether to consider reverse aspect ratios.
        variances: List of variances for x, y, w, h.
        clip: Whether to clip the prior's coordinates
            such that they are within [0, 1].

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)`

    # Output shape
        3D tensor with shape:
        (samples, num_boxes, 8)

    # References
        https://arxiv.org/abs/1512.02325
    """
    pass