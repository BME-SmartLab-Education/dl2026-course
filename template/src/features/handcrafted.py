"""Classical image-quality features for the baseline model.

Things worth measuring: sharpness (Laplacian variance, Tenengrad), exposure
(mean, percentiles, saturated and dark pixel fractions), contrast, histogram
shape, JPEG blockiness, and the geometry of the detected page quadrilateral.

Worth thinking about: all of those describe the frame as a whole. A defect that
covers one column and leaves the rest clean moves them very little. Tile-wise
statistics on a grid are one way to give a classical model a chance at that.
"""
