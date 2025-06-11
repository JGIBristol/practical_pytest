"""
Example code for finding the points on the edge of an object
"""
import shapely
import numpy as np
from skimage import measure

def points_around_edge(
    binary_img: np.ndarray, n_points: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get equally spaced points along the boundary of a binary image.

    This performs a slight smoothing, as the contour points are on the edges of each pixel
    rather than on their corners.

    :param binary_img: Binary image.
    :param n_points: Number of points to generate.
    :return: x, y coordinates of the points.
    """
    contour_points = measure.find_contours(binary_img, fully_connected="high")[0]

    shape = shapely.LinearRing(contour_points)
    distances = np.linspace(0, 1, n_points + 1)
    pts = shapely.line_interpolate_point(shape, distances, normalized=True)

    x = [p.x for p in pts]
    y = [p.y for p in pts]

    return np.array(x), np.array(y)
