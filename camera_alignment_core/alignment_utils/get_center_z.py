import numpy as np


def get_center_z(
    img_stack: np.typing.NDArray[np.uint16],
    thresh=(0.2, 99.8),
) -> int:
    """
    Get index of center z slice by finding the slice with max. contrast value
    Parameters
    ----------
    stack           a 3D (or 2D) image

    Returns
    -------
    center_z        index of center z-slice
    """
    center_z = 0
    max_contrast = 0
    all_contrast = []
    for z in range(0, img_stack.shape[0]):
        contrast = (
            np.percentile(img_stack[z, :, :], thresh[1])
            - np.percentile(img_stack[z, :, :], thresh[0])
        ) / (np.max(img_stack[z, :, :]))
        all_contrast.append(contrast)
        if contrast > max_contrast:
            center_z = z
            max_contrast = contrast

    return center_z
