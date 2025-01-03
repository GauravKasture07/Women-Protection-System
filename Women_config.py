import cv2
import numpy as np
import time
import logging
import sys
from ultralytics import YOLO

model_name = "voilanceD.pt"
violence_model = YOLO(model_name)

YOLO_CONFIG = "C:/personal/Women Protection/yolov3.cfg"
YOLO_WEIGHTS = "C:/personal/Women Protection/yolov3.weights"
YOLO_NAMES = "C:/personal/Women Protection/coco.names"

net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CONFIG)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

with open(YOLO_NAMES, "r") as f:
    classes = [line.strip() for line in f.readlines()]

violence_counter = 0

logging.basicConfig(filename='detection_log.txt', 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def log_message(message):
    logging.info(message)
    print(message)  

def resize_frame(frame, width=440):
    height = int((frame.shape[0] / frame.shape[1]) * width)
    return cv2.resize(frame, (width, height))

def detect_full_body(frame):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == "person":
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(indices) > 0:
        indices = indices.flatten()
        people_boxes = [boxes[i] for i in indices]
    else:
        people_boxes = []
    return people_boxes

def classify_gender_full_body(frame, box):
    return "Male" if np.random.rand() > 0.5 else "Female"

def detect_violence(frame):
    results = violence_model(frame)
    violence_detected = False
    for r in results:
        if 'violence' in r.names and r.boxes:
            violence_detected = True
            break
    return violence_detected

def get_simulated_location():
    return {"latitude": "12.9715987", "longitude": "77.594566"}

def save_alert(location, filename, violence_num):
    alert_message = f"Alert {violence_num}: Violence detected at location {location['latitude']}, {location['longitude']} on {time.ctime()}"
    try:
        with open("alerts.txt", "a") as file:
            file.write(alert_message + '\n')
    except IOError as e:
        log_message(f"Error writing to alerts file: {e}")
    log_message(alert_message + f" | Photo saved: {filename}")

def capture_photo(frame, violence_num):
    filename = f"violence_detected_{violence_num}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
    try:
        cv2.imwrite(filename, frame)
        # Removed logging of photo capture
    except IOError as e:
        log_message(f"Error capturing photo: {e}")
    return filename

def detect_surrounded(frame):
    people_boxes = detect_full_body(frame)
    men_count = 0
    women_count = 0
    alone_women_count = 0
    woman_surrounded_count = 0

    women_boxes = []

    for box in people_boxes:
        x, y, w, h = box
        full_body = frame[y:y+h, x:x+w]
        gender = classify_gender_full_body(frame, box)

        if gender == "Male":
            men_count += 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            women_count += 1
            women_boxes.append(box)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)

    for woman_box in women_boxes:
        wx, wy, ww, wh = woman_box
        woman_surrounded = False
        for man_box in people_boxes:
            mx, my, mw, mh = man_box
            if wx != mx and wy != my:
                if (abs(wx - mx) < ww and abs(wy - my) < wh):
                    woman_surrounded = True
                    break

        if woman_surrounded:
            woman_surrounded_count += 1

    if women_count == 1 and men_count == 0:
        alone_women_count += 1
    
    global violence_counter
    violence_detected = detect_violence(frame)
    if violence_detected:
        violence_counter += 1
        alert_message = f"Violence detected! Incident number: {violence_counter}"
        log_message(alert_message)
        
        photo_filename = capture_photo(frame, violence_counter)
        location = get_simulated_location()
        save_alert(location, photo_filename, violence_counter)

    cv2.putText(frame, f"Men: {men_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Women: {women_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Women Surrounded: {woman_surrounded_count}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Alone Women: {alone_women_count}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    return frame

def main():
    #ip_camera_url = 'http://100.82.159.27:8080/video'
    #cap = cv2.VideoCapture(ip_camera_url)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        log_message("Error: Could not open video capture.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            log_message("Error: Failed to capture frame.")
            break

        frame = resize_frame(frame)
        
        output = detect_surrounded(frame)
        cv2.imshow("Detection", output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
