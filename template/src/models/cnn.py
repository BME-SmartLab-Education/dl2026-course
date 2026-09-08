"""Convolutional models.

Start from an ImageNet-pretrained backbone with a regression head and a sigmoid
output, then improve it. Input resolution is worth experimenting with: the
defects that matter are not all the same size on the page.

Be careful with augmentation. Flips and small translations are safe. Anything
that changes image quality, such as blur, brightness jitter or added noise,
alters the thing you are trying to predict and will corrupt the label.
"""
