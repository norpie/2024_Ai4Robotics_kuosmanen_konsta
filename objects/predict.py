import cv2
import ultralytics

# Load model
model = ultralytics.YOLO('best.pt')

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Inference
    results = model(frame)

    # Render the results on the frame
    annotated_frame = results[0].plot()

    # Display the resulting frame with annotations
    cv2.imshow('YOLOv8 Webcam', annotated_frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
