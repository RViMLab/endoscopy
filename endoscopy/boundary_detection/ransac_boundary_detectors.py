import cv2
import numpy as np
from typing import Tuple


def threePointCircle(p1: np.array, p2: np.array, p3: np.array) -> Tuple[np.array, float]:
    r"""Computes a circle, given 3 points on that circle, see https://stackoverflow.com/questions/26222525/opencv-detect-partial-circle-with-noise.

    Args:
        p1 (np.array): Point 1 on circle.
        p2 (np.array): Point 2 on circle.
        p3 (np.array): Point 3 on circle.

    Return:
        center (np.array): Center of circle
        radius (float): Radius of circle
    """
    x1, x2, x3 = p1[0], p2[0], p3[0]
    y1, y2, y3 = p1[1], p2[1], p3[1]

    denom = (2*(x1*(y2-y3) - y1*(x2-x3) + x2*y3 - x3*y2))

    if denom == 0.:
        return np.array([]), None

    center = np.array([
        ((x1*x1+y1*y1)*(y2-y3) + (x2*x2+y2*y2)*(y3-y1) + (x3*x3+y3*y3)*(y1-y2))/denom,
        ((x1*x1+y1*y1)*(x3-x2) + (x2*x2+y2*y2)*(x1-x3) + (x3*x3+y3*y3)*(x2-x1))/denom
    ])

    radius = np.sqrt((center[0]-x1)*(center[0]-x1) + (center[1]-y1)*(center[1]-y1))

    return center, radius

def buildLinearSystem(pts: np.array) -> Tuple[np.array, np.array]:
    r"""Build linear system that describes circle, for example check https://math.stackexchange.com/questions/214661/circle-least-squares-fit

    Args:
        pts (np.array): Image points of shape Nx2

    Return:
        A (np.array): Linear system matrix
        b (np.array): Offset to linear equation
    """
    A = np.stack(
        (2*pts[:, 0], 2*pts[:, 1], np.ones(pts.shape[0])), axis=1
    )
    b = np.stack(
        np.square(pts[:, 0]) + np.square(pts[:, 1])
    )

    return A, b

def solveSystem(A, b) -> Tuple[np.array, float]:
    r"""Solve linear system for center and radius, for example check https://math.stackexchange.com/questions/214661/circle-least-squares-fit

    Args:
        A (np.array): Linear system matrix
        b (np.array): Offset to linear equation

    Return:
        center (np.array): Circle's center
        radius (float): Circles radius
    """
    x, res, rank, s = np.linalg.lstsq(A, b, rcond=None)

    # solve for radius, x2 = r^2 - x0^2 - x1^2
    radius = np.sqrt(x[2] + x[0]**2 + x[1]**2)

    return x[:-1], radius

def ransacBoundaryCircle(img: np.array, th: float=10., decay: float=2., fit='analytic', n_pts: int=100, n_iter: int=100, edge='canny', kwargs: dict={'threshold1': 100, 'threshold2': 200}) -> Tuple[np.array, float]:
    r"""Finds boundary circle in an endoscopic image via the following method

        Algorithm: 
            1. Edge detection
            2. Do RANSAC circle fit (https://en.wikipedia.org/wiki/Random_sample_consensus)
                    for n_iter:
                        3. Sample n_pts maybe inliers in edge image
                        4. Fit circle to n_pts, where n_pts is set to 3 for fit == 'analytic'
                        5. Find also inliers, points that are closer than th to the circle circumference
                        6. If number of also inliers bigger than best number of also inliers, re-fit circle to maybe + also inliers
                        7. Find inliers of re-fit model, if percentage of inliers better than best percentage, save radius and center of model
                        8. Divide th by decay

    Args:
        img (np.array): Grayscale image of shape HxW (ideally binary)
        th (float): Distance to circle circumference threshold, see algorithm step 5, 7
        decay (flaot): Divides th by decay at each iteration
        n_pts (int): Points to sample in edge image, see algorithm step 3
        n_iter (int): Number of iterations to improve on found circle, see algorithm step 2
        fit (str): Fit method, 'analytic' or 'numeric', if analytic, n_pts will be set to 3. Analytic fit is faster, and numeric fit is usually more stable
        edge (str): Edge detector to use, e.g. 'canny', 'sobel'
        kwargs (dict): Keyword arguments for edge detector

    Return:
        center (np.array): Circle's center
        radius (float): Circles radius
    """

    if fit == 'analytic':
        n_pts = 3

    img = img.astype(np.uint8)

    # Step 1, edge detection
    if edge == 'canny':
        img = cv2.Canny(img, **kwargs)
    elif edge == 'sobel':
        img = cv2.Sobel(img, ddepth=cv2.CV_8U, **kwargs)
    else:
        assert 'Edge detector "{}" not supported.'.format(edge)

    # Track results
    best_center = np.array([])
    best_radius = None
    best_n_also_inliers = 0
    best_percentage = 0.

    pt_set = np.where(img > 0.)
    pt_set = np.stack((pt_set[0], pt_set[1]), axis=1)

    if pt_set.size == 0:
        return best_center, best_radius

    # Step 2, RANSAC circle fit
    for i in range(n_iter):

        # Step 3, sample maybe_inliers
        pt_set_idcs = np.arange(pt_set.shape[0])
        pt_subset_idcs = np.random.choice(pt_set_idcs, size=min(n_pts, pt_set_idcs.shape[0]),replace=False)

        if pt_subset_idcs.size == 0:
            continue

        pt_subset_mask = np.zeros(pt_set_idcs.shape, dtype=bool)
        pt_subset_mask[pt_subset_idcs] = True

        maybe_inliers = pt_set[pt_subset_idcs] + 0.5

        # Step 4, fit model
        if fit == 'analytic':
            center, radius = threePointCircle(maybe_inliers[0], maybe_inliers[1], maybe_inliers[2])
        elif fit == 'numeric':
            A, b = buildLinearSystem(maybe_inliers)
            center, radius = solveSystem(A, b)
        else:
            assert 'Fit method "{}" unknown'.format(fit)

        if radius is None:
            continue

        # Step 5, find also_inliers
        pt_subset_complement = pt_set[pt_subset_mask == False]
        distance_to_center = np.linalg.norm(center - pt_subset_complement, axis=1)
        also_inliers = pt_subset_complement[np.where(np.abs(distance_to_center - radius) < th)] + 0.5

        # Step 6, re-fit model on all inliers
        if also_inliers.size > best_n_also_inliers:
            best_n_also_inliers = also_inliers.shape[0]
            inliers = np.concatenate((maybe_inliers, also_inliers))

            if fit == 'analytic':
                center, radius = threePointCircle(maybe_inliers[0], maybe_inliers[1], maybe_inliers[2])
            elif fit == 'numeric':
                A, b = buildLinearSystem(inliers)
                center, radius = solveSystem(A, b)
            else:
                assert 'Fit method "{}" unknown'.format(fit)

            # Step 7, save best model
            distance_to_center = np.linalg.norm(center - inliers, axis=1)
            total_inliers = pt_set[np.where(np.abs(distance_to_center - radius) < th)]
            
            percentage = total_inliers.shape[0]/pt_set.shape[0]
            if percentage > best_percentage:
                best_percentage = percentage
                best_center, best_radius = center, radius

        # Step 8, decay distance threshold
        th /= decay

    return best_center, best_radius
