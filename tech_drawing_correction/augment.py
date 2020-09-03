#!/usr/bin/env python3


from argparse import ArgumentParser
from collections import defaultdict
import logging
import os
import random
from time import time

import cv2
from PIL import Image
import numpy as np


LOGGER = logging.getLogger(__name__)


def _kernel(size):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _load(path, grayscale=True):
    img = Image.open(path)
    if grayscale:
        img = img.convert("L")
    img_np = np.array(img, dtype=np.float32) / 255
    return img_np


def _brightness(img_np, add):
    if add > 0:
        return (1 - img_np) * add + img_np
    else:
        return img_np * add + img_np


def _darken(img_np, amount):
    return np.maximum(img_np * amount, 1.0)


def _close(img_np, kernel_size):
    return 1 - cv2.morphologyEx(1 - img_np, cv2.MORPH_CLOSE,
                                _kernel(kernel_size))


def _open(img_np, kernel_size):
    return 1 - cv2.morphologyEx(1 - img_np, cv2.MORPH_OPEN,
                                _kernel(kernel_size))


def _dilate(img_np, kernel_size):
    return 1 - cv2.morphologyEx(1 - img_np, cv2.MORPH_DILATE,
                                _kernel(kernel_size))


def _dilate_noisy(img_np, kernel_size, noise_factor, noise_contrast):
    dilated = cv2.morphologyEx(1 - img_np, cv2.MORPH_DILATE,
                               _kernel(kernel_size))
    noise = _contrast(_generate_noise(img_np.shape, noise_factor, 5),
                      noise_contrast)
    dilated = dilated * noise
    return np.minimum(img_np, 1 - dilated)


def _gradient(img_np, left, right, top, down):
    h, w = img_np.shape
    grad_h = np.repeat(np.linspace(left, right, w).reshape(1, -1), h, axis=0)
    grad_w = np.repeat(np.linspace(top, down, h).reshape(-1, 1), w, axis=1)
    grad =  0.5 * grad_h + 0.5 * grad_w
    return grad * img_np


def _save(img_np, path):
    Image.fromarray(img_np * 255).convert("RGB").save(path)


def _generate_noise(size, factor, blur_kernel_size, normalize=True):
    noise = np.random.random(tuple(round(x / factor) for x in size))
    if blur_kernel_size is not None:
        noise = cv2.GaussianBlur(noise, (blur_kernel_size, blur_kernel_size),
                                 blur_kernel_size / 2.5)
    noise = cv2.resize(noise, tuple(reversed(size)))
    if normalize:
        noise = noise - noise.min()
        noise /= noise.max()

    return noise


def _wreck_lines(img, noise_factor, noise_contrast, noise_alpha):
    noise = _contrast(_generate_noise(img.shape, noise_factor, 5),
                      noise_contrast)
    return 1 - ((1 - noise * noise_alpha) * (1 - img))


def _noisy_open(img, noise_factor, noise_contrast):
    img_open = _open(img, 3)
    img_open[img_open < 0.9] = 0.0
    thin_lines = img_open - img
    noise = _contrast(_generate_noise(img.shape, noise_factor, 5),
                      noise_contrast)

    return img + noise * thin_lines


def _contrast(img, factor):
    return np.minimum(np.maximum(0, (img - 0.5) * factor + 0.5), 1)


def _add_noise(img, factor, alpha):
    noise = _generate_noise(img.shape, factor, 3)
    return (1 - alpha) * img + alpha * noise


def _blur(img_np, kernel):
    return cv2.GaussianBlur(img_np, (kernel, kernel), kernel / 2.5)


class Choice:

    def __init__(self, options):
        self._options = list(options)

    def __call__(self):
        return random.choice(self._options)


class Uniform:

    def __init__(self, lower, upper):
        assert upper > lower
        self._lower = lower
        self._span = upper - lower

    def __call__(self):
        return random.random() * self._span + self._lower


class RandomOp:

    def __init__(self, op, *args, **kwargs):
        self._op = op
        self._args = args
        self._kwargs = kwargs

    def __call__(self, img):
        args = [arg() for arg in self._args]
        kwargs = {k: v() for k, v in self._kwargs.items()}
        LOGGER.debug("\tApplying %s with args: %r, kwargs: %r",
                     self._op.__name__, args, kwargs)
        return self._op(img, *args, **kwargs)


class RandomApply:

    def __init__(self, *ops_and_probs):
        self._ops, probs = zip(*ops_and_probs)
        self._probs = [float(x) / sum(probs) for x in probs]

    def __call__(self, img):
        op = self._ops[random.choices(range(len(self._ops)),
                                      self._probs)[0]]
        if op is None:
            return img
        else:
            return op(img)


class GradientArgs:

    def __init__(self):
        self._rand = Uniform(0.5, 0.8)

    def __call__(self):
        a = [self._rand(), 1.0]
        b = [self._rand(), 1.0]
        random.shuffle(a)
        random.shuffle(b)
        return a + b


_DILATE_NOISY = RandomOp(_dilate_noisy,
                         kernel_size=Choice(range(2, 5)),
                         noise_factor=Choice(range(2, 9)),
                         noise_contrast=Choice(range(2, 5)))
_WRECK_LINES = RandomOp(_wreck_lines,
                        noise_factor=Choice(range(2, 9)),
                        noise_contrast=Choice(range(1, 5)),
                        noise_alpha=Uniform(0.5, 1.0))
_NOISY_OPEN = RandomOp(_noisy_open,
                       noise_factor=Choice(range(1, 5)),
                       noise_contrast=Choice(range(2, 5)))
_BLUR = RandomOp(_blur, kernel=Choice([3, 5, 7]))
_NOISE = RandomOp(_add_noise,
                 factor=Choice(range(2, 10)),
                 alpha=Uniform(0.0, 0.4))
_GRADIENT = RandomOp(lambda img, args: _gradient(img, *args),
                    GradientArgs())
_BRIGHTNESS = RandomOp(_brightness, add=Uniform(-0.5, 0.5))
_CONTRAST = RandomOp(_contrast, factor=Uniform(0.4, 1.5))


_RANDOM_PROCESS = [
        RandomApply((_DILATE_NOISY, 1), (_WRECK_LINES, 1), (_NOISY_OPEN, 1),
                    (None, 1)),
        RandomApply((_BLUR, 1), (None, 3)),
        RandomApply((_NOISE, 1), (None, 3)),
        RandomApply((_GRADIENT, 1), (None, 2)),
        RandomApply((_BRIGHTNESS, 1), (None, 2)),
        RandomApply((_CONTRAST, 1), (None, 2))
    ]


def random_augment(img):
    for random_apply in _RANDOM_PROCESS:
        img = random_apply(img)
    return img


def main():
    argp = ArgumentParser()
    argp.add_argument("test_img_path")
    argp.add_argument("--output-dir", default="../data/aug_test/")
    argp.add_argument("--test-imgs", type=int, default=20)
    argp.add_argument("--skip-random", action="store_true")
    argp.add_argument("--logging", choices=("INFO", "DEBUG"), default="INFO")
    args = argp.parse_args()

    logging.basicConfig(level=args.logging)

    os.makedirs(args.output_dir, exist_ok=True)

    img = _load(args.test_img_path)

    if not args.skip_random:
        for i in range(args.test_imgs):
            LOGGER.debug("Random augmenting img %d", i)
            _save(random_augment(img), os.path.join(args.output_dir,
                                                    "random_%03d.png" % i))

    for op, op_args, name in (
            (_brightness, (0.5,), "lighten"),
            (_brightness, (-0.4,), "darken"),
            (_contrast, (0.4,), "contrast"),
            (_close, (5,), "close"),
            (_dilate, (3,), "dillate"),
            (_dilate_noisy, (4, 2, 3), "dillate_noisy"),
            (_open, (3,), "open"),
            (_blur, (7,), "blur_7"),
            (_noisy_open, (4, 3), "noisy_open"),
            (_wreck_lines, (2, 2, 0.57), "wreck_lines"),
            (_add_noise, (10, 0.4), "add_noise_40_percent"),
            (_gradient, (0.5, 1.0, 0.75, 1.0), "gradient")):
        t0 = time()
        processed_img = op(img, *op_args)
        LOGGER.debug("Op %r, time: %.3f", op.__name__, time() - t0)
        _save(processed_img, os.path.join(args.output_dir, name + ".png"))


if __name__ == "__main__":
    main()
