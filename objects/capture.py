# Capture frames from camera and save them to disk

import cv2
import os
import sys

# Get a single arguments, name of the object/output dir
if len(sys.argv) != 2:
    print('Usage: python capture.py <object_name>')
    sys.exit(1)
object_name = sys.argv[1]

# Create a directory to save the images
if not os.path.exists('images'):
    os.makedirs('images')
if not os.path.exists('images/' + object_name):
    os.makedirs('images/' + object_name)

# Open the camera
cap = cv2.VideoCapture(0)

# Capture frames
frames = []

while True:
    ret, frame = cap.read()
    if not ret:
        print('Failed to capture frame')
        break

    cv2.imshow('frame', frame)
    frames.append(frame)
    print('Captured frame', len(frames))

    if len(frames) == 200:
        break

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Release the camera
cap.release()
cv2.destroyAllWindows()

# Save the frames
for i, frame in enumerate(frames):
    cv2.imwrite('images/' + object_name + '/' + str(i) + '.png', frame)
    print('Saved images/' + object_name + '/' + str(i) + '.png')
