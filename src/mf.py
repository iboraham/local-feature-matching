from sklearn.metrics.pairwise import euclidean_distances
import numpy as np


def match_features(features1, features2, x1, y1, x2, y2, threshold=1):
    """

    Args:
    -   features1: A numpy array of shape (n,feat_dim) representing one set of
            features, where feat_dim denotes the feature dimensionality
    -   features2: A numpy array of shape (m,feat_dim) representing a second set
            features (m not necessarily equal to n)
    -   x1: A numpy array of shape (n,) containing the x-locations of features1
    -   y1: A numpy array of shape (n,) containing the y-locations of features1
    -   x2: A numpy array of shape (m,) containing the x-locations of features2
    -   y2: A numpy array of shape (m,) containing the y-locations of features2
    -   these are provisional arguments, please feel free to change if you need

    Returns:
    -   matches: A numpy array of shape (k,2), where k is the number of matches.
            The first column is an index in features1, and the second column is
            an index in features2
    -   confidences: A numpy array of shape (k,) with the real valued confidence for
            every match
    """

    # Calculate euclidean distances

    distances = euclidean_distances(features1, features2)

    # sort distances ascending
    indices = np.argsort(distances, axis=1)
    sorted_distances = np.take_along_axis(distances, indices, axis=1)

    # Find the nearest neighbor ratios
    scores = sorted_distances[:, 0] / sorted_distances[:, 1]

    # where scores < threshold
    idx = scores < threshold
    confidences = 1 / scores[idx]

    # matches for which we have score or distance < threshold

    matches = np.zeros((confidences.shape[0], 2), dtype=int)
    matches[:, 0] = np.where(idx)[0]
    matches[:, 1] = indices[idx, 0]

    # arrange the matches and the confidences in descending order
    idx = (-confidences).argsort()
    matches = matches[idx, :]
    confidences = confidences[idx]

    return matches, confidences
