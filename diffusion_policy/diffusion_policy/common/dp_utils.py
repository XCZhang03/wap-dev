import os
import pathlib
import hydra
import torch
import dill
import numpy as np
from PIL import Image

from diffusion_policy.common.libero_utils import LANG_EMBED_CACHE_FILE
from diffusion_policy.workspace.base_workspace import BaseWorkspace

def load_checkpoint(checkpoint: str):
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=None)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    policy.eval()

    return policy, cfg


def embed_lang(instruction: str, compute_embed=False) -> np.ndarray:
    lang_embed_cache = dict(np.load(LANG_EMBED_CACHE_FILE)) if os.path.exists(LANG_EMBED_CACHE_FILE) else dict()
    lang_embed = lang_embed_cache.get(instruction, None)
    if lang_embed is not None:
        print('Loaded language embed from cache.')
    if lang_embed is None:
        if not compute_embed:
            raise ValueError(f"Language instruction not found in cache. Set compute_embed=True to compute and cache the language embedding.")
        from diffusion_policy.model.vision.model_getter import get_language_model
        lang_encode_fn = get_language_model()
        lang_embed = lang_encode_fn(instruction).astype(np.float32)
        lang_embed_cache[instruction] = lang_embed
        np.savez_compressed(LANG_EMBED_CACHE_FILE, **lang_embed_cache)
    return lang_embed


def convert_to_uint8(img: np.ndarray) -> np.ndarray:
    """Converts an image to uint8 if it is a float image.

    This is important for reducing the size of the image when sending it over the network.
    """
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    return img


def resize_with_pad(images: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
    """Replicates tf.image.resize_with_pad for multiple images using PIL. Resizes a batch of images to a target height.

    Args:
        images: A batch of images in [..., height, width, channel] format.
        height: The target height of the image.
        width: The target width of the image.
        method: The interpolation method to use. Default is bilinear.

    Returns:
        The resized images in [..., height, width, channel].
    """
    # If the images are already the correct size, return them as is.
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape

    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack([_resize_with_pad_pil(Image.fromarray(im), height, width, method=method) for im in images])
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def _resize_with_pad_pil(image: Image.Image, height: int, width: int, method: int) -> Image.Image:
    """Replicates tf.image.resize_with_pad for one image using PIL. Resizes an image to a target height and
    width without distortion by padding with zeros.

    Unlike the jax version, note that PIL uses [width, height, channel] ordering instead of [batch, h, w, c].
    """
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return image  # No need to resize if the image is already the correct size.

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    assert zero_image.size == (width, height)
    return zero_image


