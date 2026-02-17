import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from deepface import DeepFace
import threading
import time

# --- Configuration ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
DB_PATH = "known_faces"
MODEL_PATH = 'blaze_face_short_range.tflite'

# --- Shared Variables for Threading ---
current_name = "Scanning..."
is_recognizing = False  # Lock to prevent multiple recognition threads

def recognize_face_async(face_crop):
    """Background task to identify the face."""
    global current_name, is_recognizing
    try:
        # enforce_detection=False is CRITICAL for speed since we already cropped
        results = DeepFace.find(img_path=face_crop, 
                                db_path=DB_PATH, 
                                enforce_detection=False, 
                                model_name="ArcFace", 
                                silent=True)
        
        if len(results) > 0 and not results[0].empty:
            new_name = os.path.basename(results[0]['identity'][0]).split('.')[0]
            current_name = new_name
        else:
            current_name = "Unknown"
    except Exception:
        pass
    finally:
        is_recognizing = False

# --- Initialize MediaPipe ---
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

cap = cv2.VideoCapture(0)
cv2.namedWindow('Smooth Face Recognition', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Smooth Face Recognition', 800, 900)
prev_time = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # 1. High-Speed Detection (Runs at 60 FPS)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    detection_result = detector.detect(mp_image)

    if detection_result.detections:
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height

            # 2. Trigger Recognition Thread (Doesn't block the video)
            if not is_recognizing:
                face_roi = frame[max(0, y):y+h, max(0, x):x+w].copy()
                if face_roi.size > 0:
                    is_recognizing = True
                    # Start the background 'brain'
                    thread = threading.Thread(target=recognize_face_async, args=(face_roi,))
                    thread.daemon = True # Closes if main program closes
                    thread.start()

            # 3. Draw UI (Updates instantly)
            color = (0, 255, 0) if current_name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, current_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # FPS Counter
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow('Smooth Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()