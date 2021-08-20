import numpy as np
import cv2
from tqdm import tqdm
from scipy.ndimage import maximum_filter


def get_interest_points(image, feature_width=24, ksize=(3, 3), sigma=0.5):
    """
    Implementation of the Harris corner detector (See Szeliski 4.1.1) to start with.

    Args:
    -   image: A numpy array of shape (m, n, c), image may be grayscale or color
            (your choice)
    -   feature_width: integer representing the local feature width in pixels.
    -   these are provisional arguments, please feel free to change if you need.

    Returns:
    -   x: A numpy array of shape (N,) containing x-coordinates of interest points
    -   y: A numpy array of shape (N,) containing y-coordinates of interest points
    -   confidences (optional): numpy nd-array of dim (N,) containing the strength
            of each interest point
    -   scales (optional): A numpy array of shape (N,) containing the scale at each
            interest point
    -   orientations (optional): A numpy array of shape (N,) containing the orientation
            at each interest point

    See:
        https://www.microsoft.com/en-us/research/wp-content/uploads/2005/06/cvpr05.pdf
        or
        https://www.cs.ucsb.edu/~holl/pubs/Gauglitz-2011-ICIP.pdf
    """
    confidences, scales, orientations = None, None, None

    # Get the interesting points with confidences
    def get_points(image, feature_width):
        assert image.ndim == 2, "Image must be grayscale"

        # Hyperparameters
        k = 0.05
        c_robust = 0.90
        n = 1500

        # Gray filter
        # gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Apply Gaussian filter on gray image
        img = cv2.GaussianBlur(image, ksize, sigma)

        sobelX = np.array(([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]), dtype=np.float)
        sobelY = np.array(([-1, -2, -1], [0, 0, 0], [1, 2, 1]), dtype=np.float)

        # convolution
        I_x = cv2.filter2D(img, -1, sobelX)
        I_y = cv2.filter2D(img, -1, sobelY)

        # gradient covariances
        I_x_x = cv2.GaussianBlur(I_x * I_x, ksize, sigma)
        I_y_y = cv2.GaussianBlur(I_y * I_y, ksize, sigma)
        I_x_y = cv2.GaussianBlur(I_x * I_y, ksize, sigma)

        # determinant and trace
        detA = I_x_x * I_y_y - I_x_y ** 2
        traceA = I_x_x + I_y_y
        R = detA - k * traceA ** 2

        # Maximum filter by 3x3 window other values 0
        mx = maximum_filter(R, size=3)
        R_filtered = np.where(mx == R, R, 0)

        # Values greater than non-zero mean stays same, ow 0
        _R = np.where(
            R_filtered > np.true_divide(R_filtered.sum(), (R_filtered != 0).sum()),
            R_filtered,
            0,
        )
        yd, xd = np.where(_R > 0)

        # Sort candidate corners by Harris scores
        R_candids = np.array([_R[y, x] for y, x in zip(yd, xd)])
        corner_candid_indexes = [
            (y, x)
            for y, x, _ in sorted(
                zip(yd, xd, R_candids), key=lambda x: x[2], reverse=True
            )
        ]
        corner_candid_indexes = np.array(corner_candid_indexes)

        # Sort R
        R_sorted = np.sort(R_candids)[::-1]

        # Set parameters
        radiis, comparison_indexes = [], []

        # Robust
        for i, index in enumerate(corner_candid_indexes):
            robustness = c_robust * R_sorted[i]
            ind = np.argmin(R_sorted[R_sorted > robustness])
            if i != 0:
                comparison_indexes.append(corner_candid_indexes[:ind])
            else:
                comparison_indexes.append(corner_candid_indexes[:i])

        # Iterate through each corner candid and their comparisons and calculate difference
        dif = []
        for x, b in zip(corner_candid_indexes, comparison_indexes):
            difference = x - b

            no = len(corner_candid_indexes) - len(difference)
            t = np.array([99999, 99999] * no)
            t = t.reshape(no, 2)
            difference = np.vstack((x - b, t))

            dif.append(difference)
            del t, no, difference
        dif = np.array(dif)  # Convert dif to numpy array
        distance = np.hypot(
            dif[:, :, 0], dif[:, :, 1]
        )  # Calculate distance from differences

        radiis = distance.min(axis=1)  # Find minimum radius for each candid

        del distance, comparison_indexes

        # Sort candidate corners with their radiis'
        sorted_indexes = [
            x
            for x, _ in sorted(
                zip(corner_candid_indexes, radiis), key=lambda x: x[1], reverse=True
            )
        ]

        y = np.array([y for y, x in sorted_indexes[:n]])
        x = np.array([x for y, x in sorted_indexes[:n]])
        return x, y, R[y, x]

    x, y, confidences = get_points(image, feature_width)
    sum = 0
    for i in range(len(x)):
        for j in range(1, len(x)):
            sum = sum + ((x[i] - x[j])) ** 2 + ((y[i] - y[j]) ** 2) ** (1 / 2)
    scales = (len(x) ^ 2) / sum

    return x, y, confidences, scales, orientations
