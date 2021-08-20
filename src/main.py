from utils import (
    im2single,
    rgb2gray,
    cheat_interest_points,
    evaluate_correspondence,
    get_data,
)
from gf import get_features
from mf import match_features
from gip import get_interest_points
import matplotlib.pyplot as plt
import cv2
import argparse

# Test Function
def test(
    image1,
    image2,
    scale_factor,
    feature_width,
    ksize,
    sigma,
    regularizer,
    threshold,
    eval_file,
):
    image1 = cv2.resize(
        image1, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR
    )
    image2 = cv2.resize(
        image2, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR
    )
    image1_bw = rgb2gray(image1)
    image2_bw = rgb2gray(image2)
    # x1, y1, x2, y2 = cheat_interest_points(eval_file, scale_factor)
    scales1 = None
    scales2 = None
    x1, y1, _, scales1, _ = get_interest_points(image1_bw, feature_width, ksize, sigma)
    x2, y2, _, scales2, _ = get_interest_points(image2_bw, feature_width, ksize, sigma)
    image1_features = get_features(
        image1_bw, x1, y1, feature_width, sigma, ksize, regularizer, scales1
    )
    image2_features = get_features(
        image2_bw, x2, y2, feature_width, sigma, ksize, regularizer, scales2
    )
    matches, confidences = match_features(
        image1_features, image2_features, x1, y1, x2, y2, threshold
    )
    num_pts_to_evaluate = 100
    acc, c = evaluate_correspondence(
        image1,
        image2,
        eval_file,
        scale_factor,
        x1[matches[:num_pts_to_evaluate, 0]],
        y1[matches[:num_pts_to_evaluate, 0]],
        x2[matches[:num_pts_to_evaluate, 1]],
        y2[matches[:num_pts_to_evaluate, 1]],
    )
    plt.figure(figsize=(20, 10))
    plt.imshow(c)
    plt.tight_layout()
    plt.savefig("result.png")
    plt.show()
    return acc


def main(args):
    fl = args.image_name
    get_data()

    if fl == "nd":
        # Notre Dame with best performance hyperparameters --
        image1 = im2single(
            cv2.cvtColor(
                cv2.imread("paired_data/Notre Dame/921919841_a30df938f2_o.jpg"),
                cv2.COLOR_BGR2RGB,
            )
        )
        image2 = im2single(
            cv2.cvtColor(
                cv2.imread("paired_data/Notre Dame/4191453057_c86028ce1f_o.jpg"),
                cv2.COLOR_BGR2RGB,
            )
        )
        eval_file = "paired_data/Notre Dame/921919841_a30df938f2_o_to_4191453057_c86028ce1f_o.pkl"
        threshold = 1
        feature_width = 16
        scale_factor = 0.25
        sigma = 3
        ksize = (3, 3)
        regularizer = False
    elif fl == "eg":
        # Episcopal Gaudi with best performance hyperparameters --
        image1 = im2single(
            cv2.cvtColor(
                cv2.imread("paired_data/Episcopal Gaudi/4386465943_8cf9776378_o.jpg"),
                cv2.COLOR_BGR2RGB,
            )
        )
        image2 = im2single(
            cv2.cvtColor(
                cv2.imread("paired_data/Episcopal Gaudi/3743214471_1b5bbfda98_o.jpg"),
                cv2.COLOR_BGR2RGB,
            )
        )
        eval_file = "paired_data/Episcopal Gaudi/4386465943_8cf9776378_o_to_3743214471_1b5bbfda98_o.pkl"
        threshold = 1
        feature_width = 44
        scale_factor = 0.3
        sigma = 0.5
        ksize = (5, 5)
        regularizer = True
    elif fl == "mt":
        # Mt Rushmore with best performance hyperparameters -- 99%
        image1 = im2single(
            cv2.cvtColor(
                cv2.imread("paired_data/Mount Rushmore/9021235130_7c2acd9554_o.jpg"),
                cv2.COLOR_BGR2RGB,
            )
        )
        image2 = im2single(
            cv2.cvtColor(
                cv2.imread("paired_data/Mount Rushmore/9318872612_a255c874fb_o.jpg"),
                cv2.COLOR_BGR2RGB,
            )
        )
        eval_file = "paired_data/Mount Rushmore/9021235130_7c2acd9554_o_to_9318872612_a255c874fb_o.pkl"
        threshold = 1
        feature_width = 36
        scale_factor = 0.25
        sigma = 5
        ksize = (5, 5)
        regularizer = False
    else:
        raise AttributeError("File name should be one of the 'nd', 'eg' or 'mt'!")

    test(
        image1,
        image2,
        scale_factor,
        feature_width,
        ksize,
        sigma,
        regularizer,
        threshold,
        eval_file,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image-name", default="nd")
    args = parser.parse_args()
    main(args)
