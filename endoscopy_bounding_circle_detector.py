import cv2
import numpy as np
from typing import Tuple


class EndoscopyBoundingCircleDetector():
    def __init__(self, buffer_size: int=1):
        """Bounding circle detection for endoscopic images.

        Args:
            buffer_size (int): Optional argument to buffer past seen images

        Example:
            ebcd = EndoscopyBoundingCircleDetector(buffer_size=1)

            center, radius = ebcd.findBoundingCircle(img, th1=5, th2=200, th3=10, n_pts=100, n_iter=3)
            center, radius = center.astype(np.int), int(radius)

            cv2.circle(img, (center[1], center[0]), radius, (0,255,255))
            cv2.circle(img, (center[1], center[0]), 2, (255,0,255), 4)

            cv2.imshow('img', img)
            cv2.waitKey()
        """
        self.buffer_size = buffer_size
        self.buffer = []
        self.last_center = np.array([])
        self.last_radius = None

    def findBoundingCircle(self, img: np.array, th1: int=5. , th2: int=200., th3: float=10., decay: float=2., n_pts: int=100, n_iter: int=3) -> Tuple[np.array, float]:
        """Finds bounding circle in an endoscopic image via the following method

            Algorithm: 
                1. Turn image into grayscale and whiten where img > th1
                2. Sobel filter edge detection
                3. Compute moving average of length self.buffer_size of edge image and blacken where moving average < th2
                4. Update linear circle fit under outlier detection
                       for n_iter:
                           5. Sample n_pts points in edge image and discard point where distance to circle > th3
                           6. Minimize linear circle equation

        Args:
            img (np.array): Image of shape CxHxW
            th1 (int): Whiten threshold, see algorithm step 1
            th2 (int): Moving avergage blacken threshold, see algorithm step 2
            th3 (float): Distance to circle discard threshold, see algorithm step 5
            decay (flaot): Divides th3 by decay at each iteration
            n_pts (int): Points to sample in edge image, see algorithm step 5
            n_iter (int): Number of iterations to improve on found circle, see algorithm step 4

        Return:
            center (np.array): Circle's center
            radius (float): Circles radius
        """
        # Step 1, grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.where(img < th1, 0, 255).astype(np.uint8)

        # Step 2, Sobel edge detection
        img = cv2.Sobel(img, cv2.CV_8U, 1, 1)

        # Step 3, moving average
        self.buffer.append(img)

        if len(self.buffer) == self.buffer_size:
            avg = np.array(self.buffer)
            avg = avg.mean(axis=0)
            avg = np.where(avg < th2, 0, 255)
            self.buffer.pop(0)
            edges = np.where(avg > 0)

            # Step 4, iterate circle fit
            for _ in range(n_iter):

                # Step 5, sample
                idcs = np.random.choice(np.arange(edges[0].size), size=n_pts,replace=False)
                pts = np.stack((edges[0][idcs], edges[1][idcs]), axis=1) + 0.5

                # Step 5, remove outliers, prior to fit
                if self.last_center.size is not 0 and self.last_radius:
                    distance_to_center = np.linalg.norm(self.last_center - pts, axis=1)
                    del_idx = np.where(np.abs(self.last_radius - distance_to_center) > th3)
                    pts = np.delete(pts, del_idx, axis=0).reshape(-1, 2)
                    th3 = th3/decay

                # Step 6, fit
                A, b = self._buildLinearSystem(pts)
                self.last_center, self.last_radius = self._solveSystem(A, b)

        return self.last_center, self.last_radius

    def _buildLinearSystem(self, pts: np.array) -> Tuple[np.array, np.array]:
        """Build linear system that describes circle, for example check https://math.stackexchange.com/questions/214661/circle-least-squares-fit

        Args:

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

    def _solveSystem(self, A, b) -> Tuple[np.array, float]:
        """Solve linear system for center and radius, for example check https://math.stackexchange.com/questions/214661/circle-least-squares-fit

        Args:

        Return:
            center (np.array): Circle's center
            radius (float): Circles radius
        """
        x, res, rank, s = np.linalg.lstsq(A, b, rcond=None)

        # solve for radius, x2 = r^2 - x0^2 - x1^2
        radius = np.sqrt(x[2] + x[0]**2 + x[1]**2)

        return x[:-1], radius


def threePointCircle(p1: np.array, p2: np.array, p3: np.array) -> Tuple[np.array, float]:
    """Computes a circle, given 3 points on that circle, see https://stackoverflow.com/questions/26222525/opencv-detect-partial-circle-with-noise.

    Args:
        p1 (np.array): Point 1 on circle in OpenCV convention.
        p2 (np.array): Point 2 on circle in OpenCV convention.
        p3 (np.array): Point 3 on circle in OpenCV convention.

    Return:
        center (np.array): Center of circle
        radius (float): Radius of circle
    """
    x1, x2, x3 = p1[0], p2[0], p3[0]
    y1, y2, y3 = p1[1], p2[1], p3[1]

    center = np.array([
        ((x1*x1+y1*y1)*(y2-y3) + (x2*x2+y2*y2)*(y3-y1) + (x3*x3+y3*y3)*(y1-y2))/(2*(x1*(y2-y3) - y1*(x2-x3) + x2*y3 - x3*y2)),
        ((x1*x1+y1*y1)*(x3-x2) + (x2*x2+y2*y2)*(x1-x3) + (x3*x3+y3*y3)*(x2-x1))/(2*(x1*(y2-y3) - y1*(x2-x3) + x2*y3 - x3*y2))
    ])

    radius = np.sqrt((center[0]-x1)*(center[0]-x1) + (center[1]-y1)*(center[1]-y1))

    return center, radius


if __name__ == '__main__':
    import os

    prefix = os.getcwd()
    file = 'sample.mp4'

    vr = cv2.VideoCapture(os.path.join(prefix, file))

    ebcd = EndoscopyBoundingCircleDetector(buffer_size=1)
 
    while vr.isOpened():

        _, img = vr.read()
        if img is None:
            break
        img = cv2.resize(img, (640, 360))

        center, radius = ebcd.findBoundingCircle(img, th1=5, th2=200, th3=10, decay=2., n_pts=100, n_iter=4)
        center, radius = center.astype(np.int), int(radius)

        cv2.circle(img, (center[1], center[0]), radius, (0,255,255))
        cv2.circle(img, (center[1], center[0]), 2, (255,0,255), 4)

        cv2.imshow('img', img)
        cv2.waitKey()
