from tqdm import tqdm
from sklearn.preprocessing import minmax_scale
import numpy as np
import cv2


def get_features(image, x, y, feature_width, sigma, ksize, regularizer, scales=None):
    """

    Args:
    -   image: A numpy array of shape (m,n) or (m,n,c). can be grayscale or color, your choice
    -   x: A numpy array of shape (k,), the x-coordinates of interest points
    -   y: A numpy array of shape (k,), the y-coordinates of interest points
    -   feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e. every
                cell of your local SIFT-like feature will have an integer width
                and height). This is the initial window size you examine around
                each keypoint.
    -   scales: Python list or tuple if you want to detect and describe features
            at multiple scales
    -   these are provisional arguments, please feel free to change if you need.

    You may also detect and describe features at particular orientations.

    Returns:
    -   fv: A numpy array of shape (k, feat_dim) representing a feature vector.
            "feat_dim" is the feature_dimensionality (e.g. 128 for standard SIFT).
            These are the computed features.
    """
    assert image.ndim == 2, "Image must be grayscale"

    # Check value is out of the bonds
    def check_out(xi, win_rng, yi):
        a = xi + win_rng
        b = yi + win_rng
        c = yi - win_rng
        d = xi - win_rng
        limx, limy = image.shape
        return a > limx or b > limy or c < 0 or d < 0

    # Create histogram by angles and magnitudes array and window size
    def create_histogram(ap, mp, min, max, w=4):
        histograms = []
        for i in range(0, feature_width, w):
            for j in range(0, feature_width, w):
                apb = ap[i : i + w, j : j + w]
                mpb = mp[i : i + w, j : j + w]
                hist, _ = np.histogram(apb, range=(min, max), bins=8, weights=mpb)
                histograms.append(hist)
        return np.array(histograms).reshape(
            -1,
        )

    def get_features_scale(image, x, y, feature_width):
        # Gaussian smoothing
        img = cv2.GaussianBlur(image, ksize, sigma)
        # img=cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT)

        # Ix and Iy
        I_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize)
        I_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize)

        # allocate memory
        fv = np.zeros((x.shape[0], int(feature_width ** 2 / 2)))

        # Angles and magnitudes
        angles = np.arctan2(I_y, I_x)
        min, max = angles.min(), angles.max()
        magnitudes = np.sqrt(I_x ** 2 + I_y ** 2)

        # half of feature width
        win_rng = feature_width // 2

        # go through each key point
        for i in range(x.shape[0]):
            xi = int(x[i])
            yi = int(y[i])
            if check_out(xi, win_rng, yi):
                continue
            else:
                # get patch
                angles_patch = angles[
                    yi - win_rng : yi + win_rng, xi - win_rng : xi + win_rng
                ]
                if regularizer:
                    angles_patch = cv2.GaussianBlur(angles_patch, ksize, sigma)

                magnitudes_patch = magnitudes[
                    yi - win_rng : yi + win_rng, xi - win_rng : xi + win_rng
                ]
                if regularizer:
                    magnitudes_patch = cv2.GaussianBlur(magnitudes_patch, ksize, sigma)

                # write on feature vector
                hist = create_histogram(angles_patch, magnitudes_patch, min, max)
                if regularizer:
                    fv[i, :] = minmax_scale(hist)
                else:
                    fv[i, :] = hist

        fv = fv ** 0.5
        return fv

    fv = get_features_scale(image, x, y, feature_width)
    return fv
