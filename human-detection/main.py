from moviepy.editor import ImageSequenceClip
import cv2
import os

# Models
modelFile = os.path.join(
    "models", "ssd_mobilenet_v2_coco_2018_03_29", "frozen_inference_graph.pb"
)
configFile = os.path.join("models", "ssd_mobilenet_v2_coco_2018_03_29.pbtxt")
# Read the Tensorflow network
net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

# Read the class labels
classFile = "coco_class_labels.txt"
with open(classFile) as fp:
    labels = fp.read().split("\n")

# For ach file in the directory
def detect_objects(net, im, dim=300):

    # Create a blob from the image
    blob = cv2.dnn.blobFromImage(
        im, 1.0, size=(dim, dim), mean=(0, 0, 0), swapRB=True, crop=False
    )

    # Pass blob to the network
    net.setInput(blob)

    # Peform Prediction
    objects = net.forward()
    return objects


FONTFACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1


def display_text(im, text, x, y):
    # Get text size
    textSize = cv2.getTextSize(text, FONTFACE, FONT_SCALE, THICKNESS)
    dim = textSize[0]
    baseline = textSize[1]

    # Use text size to create a black rectangle
    cv2.rectangle(
        im,
        (x, y - dim[1] - baseline),
        (x + dim[0], y + baseline),
        (0, 0, 0),
        cv2.FILLED,
    )

    # Display text inside the rectangle
    cv2.putText(
        im,
        text,
        (x, y - 5),
        FONTFACE,
        FONT_SCALE,
        (0, 255, 255),
        THICKNESS,
        cv2.LINE_AA,
    )
    return im


def display_objects(im, humans):
    rows = im.shape[0]
    cols = im.shape[1]

    # For every Detected Object
    for i in range(len(humans)):
        # Find the class and confidence
        classId = int(objects[0, 0, i, 1])
        score = float(objects[0, 0, i, 2])

        # Recover original cordinates from normalized coordinates
        x = int(objects[0, 0, i, 3] * cols)
        y = int(objects[0, 0, i, 4] * rows)
        w = int(objects[0, 0, i, 5] * cols - x)
        h = int(objects[0, 0, i, 6] * rows - y)

        display_text(im, "{} - {}".format(labels[classId], score), x, y)
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 255, 255), 2)

    # Convert Image to RGB since we are using Matplotlib for displaying image
    mp_img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return mp_img


def filter_humans(objects, threshold=0.5):
    humans = []
    for i in range(objects.shape[2]):
        classId = int(objects[0, 0, i, 1])
        score = float(objects[0, 0, i, 2])

        if classId == 1 and score > threshold:
            humans.append(objects[0, 0, i])
    return humans


if not os.path.exists("output"):
    os.makedirs("output")

raw_vids = os.listdir("data")
vids = []
for vid in raw_vids:
    if not vid.endswith(".avi"):
        continue
    if os.path.exists("output/" + vid.replace(".avi", "") + "/video.mp4"):
        continue
    vids.append(vid)
amount = len(vids)

for i, vid in enumerate(vids):
    print(f"Processing {i+1}/{amount} - {vid}")
    cap = cv2.VideoCapture(os.path.join("data", vid))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frames = []
    if cap.isOpened() == False:
        print("Error opening video stream or file")

    video_humans = []

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            objects = detect_objects(net, frame)
            humans = filter_humans(objects, threshold=0.5)
            video_humans.append((humans, frame))
            generated = display_objects(frame, humans)
            frames.append(generated)
        else:
            break
    cap.release()
    clip = ImageSequenceClip(frames, fps=24)
    output = "output/" + vid.replace(".avi", "") + "/video.mp4"
    os.makedirs(os.path.dirname(output), exist_ok=True)
    print(f"Saving to {output}")
    clip.write_videofile(output, threads=12, codec="libx264")
    clip.close()
    if len(video_humans) == 0:
        continue
    # Save largest human in frame
    max_human_size, max_frame = 0, None
    for i, (humans, frame) in enumerate(video_humans):
        for human in humans:
            x = human[3]
            y = human[4]
            w = human[5] - x
            h = human[6] - y
            if w * h > max_human_size:
                max_human_size = w * h
                max_frame = frame
            else:
                print(f"Skipping {i} - {w*h} < {max_human_size}")
                continue
    if max_frame is None:
        continue
    cv2.imwrite("output/" + vid.replace(".avi", "") + "/largest_human.jpg", max_frame)
    max_human_size, max_frame = 0, None
    print(f"Saved largest human in {vid}")
