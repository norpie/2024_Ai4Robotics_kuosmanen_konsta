import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_drawing

import cv2

base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)


class HandParser:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.hands = mp_hands.Hands(mode, maxHands)

    def find_hands(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        hands = []
        if not results.multi_hand_landmarks:
            return hands
        for hand in results.multi_hand_landmarks:
            hands.append(hand)
        return hands

    def recognize_gesture(self, img):
        mp_img = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=img
        )
        recognition_result = recognizer.recognize(mp_img)
        if not recognition_result.gestures:
            print("No gesture detected")
            return "No gesture detected", 0.0
        if not recognition_result.gestures[0]:
            print("No gesture detected [0]")
            return "No gesture detected", 0.0
        if not recognition_result.gestures[0][0]:
            print("No gesture detected [0][0]")
            return "No gesture detected", 0.0
        top_gesture = recognition_result.gestures[0][0]
        return top_gesture.category_name, top_gesture.score


    def get_landmarks(self, img, hand):
        lmList = []
        for id, lm in enumerate(hand.landmark):
            h, w, _ = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmList.append([id, cx, cy])
        return lmList

    def get_bounds(self, img, hand):
        h, w, _ = img.shape
        xList, yList = [], []
        for lm in hand.landmark:
            xList.append(lm.x)
            yList.append(lm.y)
        xmin, xmax = min(xList), max(xList)
        ymin, ymax = min(yList), max(yList)
        bbox = xmin * w, ymin * h, xmax * w, ymax * h
        return bbox


class HandPainter:

    def draw_hand(self, img, hand, color1=(121, 22, 76), color2=(250, 44, 250)):
        mp_drawing.draw_landmarks(
            img,
            hand,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=color1, thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=color2, thickness=2, circle_radius=2),
        )
        return img

    def draw_landmarks(self, hand, img, color=(255, 0, 255)):
        for _, lm in enumerate(hand.landmark):
            h, w, _ = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 7, color, cv2.FILLED)
        return img

    def draw_bounds(self, img, bounds, text="", color=(255, 0, 255)):
        x_min, y_min, x_max, y_max = bounds
        x = int(x_max)
        y = int(y_max)
        w = int(x_min)
        h = int(y_min)
        cv2.rectangle(img, (x, y), (w, h), color, 2)
        if text != "":
            cv2.putText(
                img, text, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA
            )
        return img


def main():
    cap = cv2.VideoCapture(0)
    parser = HandParser()
    painter = HandPainter()
    while True:
        _, img = cap.read()
        hands = parser.find_hands(img)
        gesture, score = parser.recognize_gesture(img)
        for hand in hands:
            img = painter.draw_hand(img, hand)
            bounds = parser.get_bounds(img, hand)
            img = painter.draw_bounds(img, bounds, text=gesture + " " + str(score))
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
