import streamlit as st
import cv2

from ultralytics import YOLO

# Load YOLO model
model = YOLO("best.pt")


# Function to detect anomalies
def detect_anomalies(frame):
    results = model(frame)
    return results


# Streamlit app
st.title("Wood Anomaly Detection")
run = st.button("Start Camera")

if run:
    # Try different camera indices
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame to 640x640
            frame = cv2.resize(frame, (640, 640))

            # Detect anomalies
            results = detect_anomalies(frame)
            # Draw bounding boxes
            for box in results[0].boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = box.conf[0]
                cls = box.cls[0]
                label = f"{model.names[int(cls)]} {conf:.2f}"
                cv2.rectangle(
                    frame,
                    (int(xyxy[0]), int(xyxy[1])),
                    (int(xyxy[2]), int(xyxy[3])),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    label,
                    (int(xyxy[0]), int(xyxy[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
                
            # Convert the image to RGB format for Streamlit
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Display the frame in Streamlit
            stframe.image(img_rgb, channels="RGB")
        cap.release()
    else:
        st.error("No camera found. Please check your camera connection.")
