import cv2
import numpy as np
from typing import Tuple


class RansacBoundaryCircleDetector():
    def __init__(self, buffer_size: int=1):
        """Boundary circle detection for endoscopic images.

        Args:
            buffer_size (int): Optional argument to buffer past seen images

        Examples:
            Numeric and Canny edge detector:
                bcd = RansacBoundaryCircleDetector(buffer_size=1)

                # numeric fit, canny edge detector
                center, radius = bcd.findBoundaryCircle(img, th1=5, th2=100, th3=10, decay=2., fit='numeric', n_pts=10, n_iter=200)

                if radius is not None:
                    center, radius = center.astype(np.int), int(radius)

                    cv2.circle(img, (center[1], center[0]), radius, (0,255,255))
                    cv2.circle(img, (center[1], center[0]), 2, (255,0,255), 4)

                    cv2.imshow('img', img)
                    cv2.waitKey()
            
            Analytic and Sobel edge detector
                bcd = RansacBoundaryCircleDetector(buffer_size=1)

                # analytic fit, sobel edge detector
                center, radius = bcd.findBoundaryCircle(img, th1=5, th2=100, th3=10, decay=2., fit='analytic', n_pts=10, n_iter=200, edge='sobel', kwargs={'dx': 1, 'dy': 1})
                   
                if radius is not None:
                    center, radius = center.astype(np.int), int(radius)

                    cv2.circle(img, (center[1], center[0]), radius, (0,255,255))
                    cv2.circle(img, (center[1], center[0]), 2, (255,0,255), 4)

                    cv2.imshow('img', img)
                    cv2.waitKey()
        """
        self.buffer_size = buffer_size
        self.buffer = []
        self.best_center = np.array([])
        self.best_radius = None

    def findBoundaryCircle(self, img: np.array, th1: int=5. , th2: int=200., th3: float=10., decay: float=2., fit='analytic', n_pts: int=100, n_iter: int=100, edge='canny', kwargs: dict={'threshold1': 100, 'threshold2': 200}) -> Tuple[np.array, float]:
        """Finds boundary circle in an endoscopic image via the following method

            Algorithm: 
                1. Turn image into grayscale and whiten where img > th1
                2. Edge detection
                3. Compute moving average of length self.buffer_size of edge image and blacken where moving average < th2
                4. Do RANSAC circle fit (https://en.wikipedia.org/wiki/Random_sample_consensus)
                       for n_iter:
                           5.  Sample n_pts maybe inliers in edge image
                           6.  Fit circle to n_pts, where n_pts is set to 3 for fit == 'analytic'
                           7.  Find also inliers, points that are closer than th3 to the circle circumference
                           8.  If number of also inliers bigger than best number of also inliers, re-fit circle to maybe + also inliers
                           9.  Find inliers of re-fit model, if percentage of inliers better than best percentage, save radius and center of model
                           10. Divide th3 by decay

        Args:
            img (np.array): Image of shape CxHxW
            th1 (int): Whiten threshold, see algorithm step 1
            th2 (int): Moving avergage blacken threshold, see algorithm step 3
            th3 (float): Distance to circle circumference threshold, see algorithm step 7, 9
            decay (flaot): Divides th3 by decay at each iteration
            n_pts (int): Points to sample in edge image, see algorithm step 5
            n_iter (int): Number of iterations to improve on found circle, see algorithm step 4
            fit (str): Fit method, 'analytic' or 'numeric', if analytic, n_pts will be set to 3. Analytic fit is faster, and numeric fit is usually more stable

        Return:
            center (np.array): Circle's center
            radius (float): Circles radius
        """

        if fit == 'analytic':
            n_pts = 3

        # Step 1, grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.where(img < th1, 0, 255).astype(np.uint8)

        # Step 2, edge detection
        if edge == 'canny':
            img = cv2.Canny(img, **kwargs)
        elif edge == 'sobel':
            img = cv2.Sobel(img, ddepth=cv2.CV_8U, **kwargs)
        else:
            assert 'Edge detector "{}" not supported.'.format(edge)

        # Step 3, moving average
        self.buffer.append(img)

        # Track results
        best_n_also_inliers = 0
        best_percentage = 0.

        if len(self.buffer) == self.buffer_size:
            avg = np.array(self.buffer)
            avg = avg.mean(axis=0)
            pt_set = np.where(avg > th2)
            self.buffer.pop(0)
            pt_set = np.stack((pt_set[0], pt_set[1]), axis=1)

            if pt_set.size == 0:
                return self.best_center, self.best_radius

            # Step 4, RANSAC circle fit
            for i in range(n_iter):

                # Step 5, sample maybe_inliers
                pt_set_idcs = np.arange(pt_set.shape[0])
                pt_subset_idcs = np.random.choice(pt_set_idcs, size=min(n_pts, pt_set_idcs.shape[0]),replace=False)

                if pt_subset_idcs.size == 0:
                    continue

                pt_subset_mask = np.zeros(pt_set_idcs.shape, dtype=bool)
                pt_subset_mask[pt_subset_idcs] = True

                maybe_inliers = pt_set[pt_subset_idcs] + 0.5

                # Step 6, fit model
                if fit == 'analytic':
                    center, radius = self._threePointCircle(maybe_inliers[0], maybe_inliers[1], maybe_inliers[2])
                elif fit == 'numeric':
                    A, b = self._buildLinearSystem(maybe_inliers)
                    center, radius = self._solveSystem(A, b)
                else:
                    assert 'Fit method "{}" unknown'.format(fit)

                if radius is None:
                    continue

                # Step 7, find also_inliers
                pt_subset_complement = pt_set[pt_subset_mask == False]
                distance_to_center = np.linalg.norm(center - pt_subset_complement, axis=1)
                also_inliers = pt_subset_complement[np.where(np.abs(distance_to_center - radius) < th3)] + 0.5

                # Step 8, re-fit model on all inliers
                if also_inliers.size > best_n_also_inliers:
                    best_n_also_inliers = also_inliers.shape[0]
                    inliers = np.concatenate((maybe_inliers, also_inliers))

                    if fit == 'analytic':
                        center, radius = self._threePointCircle(maybe_inliers[0], maybe_inliers[1], maybe_inliers[2])
                    elif fit == 'numeric':
                        A, b = self._buildLinearSystem(inliers)
                        center, radius = self._solveSystem(A, b)
                    else:
                        assert 'Fit method "{}" unknown'.format(fit)

                    # Step 9, save best model
                    distance_to_center = np.linalg.norm(center - inliers, axis=1)
                    total_inliers = pt_set[np.where(np.abs(distance_to_center - radius) < th3)]
                    
                    percentage = total_inliers.shape[0]/pt_set.shape[0]
                    if percentage > best_percentage:
                        best_percentage = percentage
                        self.best_center, self.best_radius = center, radius

                # Step 10, decay distance threshold
                th3 /= decay

        return self.best_center, self.best_radius

    def _threePointCircle(self, p1: np.array, p2: np.array, p3: np.array) -> Tuple[np.array, float]:
        """Computes a circle, given 3 points on that circle, see https://stackoverflow.com/questions/26222525/opencv-detect-partial-circle-with-noise.

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

    def _buildLinearSystem(self, pts: np.array, ) -> Tuple[np.array, np.array]:
        """Build linear system that describes circle, for example check https://math.stackexchange.com/questions/214661/circle-least-squares-fit

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

    def _solveSystem(self, A, b) -> Tuple[np.array, float]:
        """Solve linear system for center and radius, for example check https://math.stackexchange.com/questions/214661/circle-least-squares-fit

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


if __name__ == '__main__':
    import os
    from com_boundary_detectors import boundaryRectangle

    prefix = os.getcwd()
    file = 'data/endo.mp4'

    vr = cv2.VideoCapture(os.path.join(prefix, file))

    bcd = RansacBoundaryCircleDetector(buffer_size=1)
 
    while vr.isOpened():

        _, img = vr.read()
        if img is None:
            break

        img = cv2.resize(img, (640, 360))
        img = img[5:-5,:-5,:] # remove black bottom and top rows

        top_left, shape = boundaryRectangle(img, 5)
        center, radius = bcd.findBoundaryCircle(img, th1=5, th2=100, th3=10, decay=1., fit='numeric', n_pts=100, n_iter=10)
        if radius is not None:
            top_left, shape = top_left.astype(np.int), [int(i) for i in shape]
            center, radius = center.astype(np.int), int(radius)

            cv2.rectangle(img, (top_left[1], top_left[0]), (top_left[1] + shape[1], top_left[0] + shape[0]), (255, 255, 0), 1)
            cv2.circle(img, (center[1], center[0]), radius, (0,255,255))
            cv2.circle(img, (center[1], center[0]), 2, (255,0,255), 4)

            cv2.imshow('img', img)
            cv2.waitKey(1)
