# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import math
import traceback


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def gaussian_blur(img, kernel_size=5):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)


def region_of_interest(img):
    height = img.shape[0]
    width = img.shape[1]
    vertices = np.array(
        [
            [
                (0, height),
                (width / 2, height / 2),
                (width, height),
            ]
        ],
        np.int32,
    )
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def overlay_lines(img, lines, color=[255, 0, 0], thickness=2):
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
    return img


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(
        img,
        rho,
        theta,
        threshold,
        np.array([]),
        minLineLength=min_line_len,
        maxLineGap=max_line_gap,
    )
    return lines


def weighted_img(img, initial_img, α=0.8, β=1.0, γ=0.0):
    return cv2.addWeighted(initial_img, α, img, β, γ)


def filtered_lines(height, lines):
    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)  # <-- Calculating the slope.
            if math.fabs(slope) < 0.5:  # <-- Only consider extreme slope
                continue
            if slope <= 0:  # <-- If the slope is negative, left group.
                left_line_x.extend([x1, x2])
                left_line_y.extend([y1, y2])
            else:  # <-- Otherwise, right group.
                right_line_x.extend([x1, x2])
                right_line_y.extend([y1, y2])
    min_y = int(height * (3 / 5))  # <-- Just below the horizon
    max_y = height  # <-- The bottom of the image
    poly_left = np.poly1d(np.polyfit(left_line_y, left_line_x, deg=1))
    left_x_start = int(poly_left(max_y))
    left_x_end = int(poly_left(min_y))
    polyfit_right = np.polyfit(right_line_y, right_line_x, deg=1)
    poly_right = np.poly1d(polyfit_right)
    right_x_start = int(poly_right(max_y))
    right_x_end = int(poly_right(min_y))
    return [
        [
            [left_x_start, max_y, left_x_end, min_y],
            [right_x_start, max_y, right_x_end, min_y],
        ]
    ]


def get_lines(img):
    gray = grayscale(img)
    blur_gray = gaussian_blur(gray, 5)
    edges = canny(blur_gray, 50, 150)
    masked_edges = region_of_interest(edges)
    lines = hough_lines(masked_edges, 6, np.pi / 60, 160, 40, 25)
    return filtered_lines(img.shape[0], lines)

# vids = os.listdir("vids")
# random = np.random.choice(vids)
cap = cv2.VideoCapture(f"vids/solidYellowLeft.mp4")
if cap.isOpened() == False:
    print("Error opening video stream or file")

while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        try:
            generated = overlay_lines(frame, get_lines(frame))
            cv2.imshow("Processed", generated)
        except Exception as e:
            print(traceback.format_exc())
            continue
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
