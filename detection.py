from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2
from pymongo import MongoClient
from datetime import datetime
import numpy as np

# Koneksi ke MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client.supermarket
collection = db.visitor

# Load model YOLO
model = YOLO("model/best.pt")

# Fungsi menyimpan hasil prediksi ke MongoDB
def save_to_mongodb(male_count, female_count, collection):
    data = {
        'gender': f"Female: {female_count}, Male: {male_count}",
        'days': datetime.now().strftime('%A'),
        'date': datetime.now().strftime("%Y-%m-%d"),
        'total': male_count + female_count
    }
    collection.insert_one(data)

# Parameter deteksi
confidence_threshold = 0.5
iou_threshold = 0.45

# Video capture
cap = cv2.VideoCapture('data/dataset.mp4')
assert cap.isOpened()
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Region of interest
region_of_interest = [(20, 450), (1700, 454), (1700, 410), (20, 410)]

# Video writer
video_writer = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

# Init Object Counter
counter = object_counter.ObjectCounter(model.names, view_img=True, reg_pts=region_of_interest, draw_tracks=True)

# Dictionary untuk melacak objek yang sudah terdeteksi
detected_objects = {}
next_object_id = 0

def calculate_centroid(x1, y1, x2, y2):
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def is_point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

male_count = 0
female_count = 0

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        break

    results = model.predict(source=im0, conf=confidence_threshold, iou=iou_threshold)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_index = int(box.cls[0])
            class_name = model.names[class_index]
            confidence = box.conf[0]

            # Menghitung centroid dari bounding box
            centroid = calculate_centroid(x1, y1, x2, y2)

            # Deteksi objek
            found = False
            object_id = None
            for obj_id, (c_x, c_y, track, counted) in detected_objects.items():
                if abs(centroid[0] - c_x) <= 50 and abs(centroid[1] - c_y) <= 50:
                    found = True
                    object_id = obj_id
                    detected_objects[object_id] = (centroid[0], centroid[1], track + [centroid], counted)
                    break

            if not found:
                object_id = next_object_id
                detected_objects[object_id] = (centroid[0], centroid[1], [centroid], False)
                next_object_id += 1

            if not detected_objects[object_id][3] and is_point_in_polygon(centroid, region_of_interest):
                detected_objects[object_id] = (centroid[0], centroid[1], detected_objects[object_id][2], True)
                if class_name == 'male':
                    male_count += 1
                elif class_name == 'female':
                    female_count += 1

            # Gambar ekor objek
            track = detected_objects[object_id][2]
            if len(track) > 1:
                for i in range(len(track) - 1):
                    cv2.line(im0, track[i], track[i + 1], (0, 255, 255), 2)

            cv2.circle(im0, centroid, 5, (0, 0, 255), -1)
            color = (0, 255, 0) if class_name == 'male' else (0, 0, 255)
            cv2.rectangle(im0, (x1, y1), (x2, y2), color, 2)
            cv2.putText(im0, f'ID: {object_id} {class_name} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if detected_objects[object_id][3]:
                cv2.putText(im0, 'Counted', (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.polylines(im0, [np.array(region_of_interest, np.int32)], isClosed=True, color=(255, 255, 255), thickness=2)
    cv2.putText(im0, f'Male: {male_count}, Female: {female_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    im0 = counter.start_counting(im0, results)
    video_writer.write(im0)

    # Simpan data setiap beberapa frame ke MongoDB
    if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % 30 == 0:
        save_to_mongodb(male_count, female_count, collection)

cap.release()
video_writer.release()
cv2.destroyAllWindows()