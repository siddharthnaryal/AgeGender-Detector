import os
import sys
import cv2
import numpy as np
import time
class AgeGenderDetector:
    AGE_RANGES = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    GENDER_CLASSES = ['Male', 'Female']

    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(self.base_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)

        #define model paths
        #just copy these model names and download the first link you find. put all in models file
        self.face_proto = os.path.join(self.models_dir, "deploy.prototxt")
        self.face_model = os.path.join(self.models_dir, "res10_300x300_ssd_iter_140000.caffemodel")
        self.age_proto = os.path.join(self.models_dir, "age_deploy.prototxt")
        self.age_model = os.path.join(self.models_dir, "age_net.caffemodel")
        self.gender_proto = os.path.join(self.models_dir, "gender_deploy.prototxt")
        self.gender_model = os.path.join(self.models_dir, "gender_net.caffemodel")

        self._download_verified_models()

        print("Loading models...")
        try:
            self.face_net = cv2.dnn.readNet(self.face_model, self.face_proto)
            self.age_net = cv2.dnn.readNetFromCaffe(self.age_proto, self.age_model)
            self.gender_net = cv2.dnn.readNetFromCaffe(self.gender_proto, self.gender_model)
            print("All models loaded successfully")
        except Exception as e:
            print(f"Critical error loading models: {e}")
            sys.exit(1)

    def _download_verified_models(self):
        pass

    def detect_faces(self, image):
        height, width = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
        self.face_net.setInput(blob)
        detections = self.face_net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                x1 = max(0, int(detections[0, 0, i, 3] * width))
                y1 = max(0, int(detections[0, 0, i, 4] * height))
                x2 = min(width, int(detections[0, 0, i, 5] * width))
                y2 = min(height, int(detections[0, 0, i, 6] * height))
                w, h = x2 - x1, y2 - y1
                if w > 0 and h > 0:
                    faces.append([x1, y1, w, h, confidence])
        return faces

    def predict_age(self, face_img):
        try:
            if face_img.shape[0] < 50 or face_img.shape[1] < 50:
                print("Skipped age prediction: face too small.")
                return "Unknown", 0.0

            blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227),
                                         [78.4263377603, 87.7689143744, 114.895847746],
                                         swapRB=False)
            self.age_net.setInput(blob)
            preds = self.age_net.forward()

            if preds.shape[1] != len(self.AGE_RANGES):
                print("Invalid prediction output shape for age:", preds.shape)
                return "Unknown", 0.0

            age_idx = preds[0].argmax()
            return self.AGE_RANGES[age_idx], float(preds[0][age_idx])
        except Exception as e:
            print(f"Age prediction failed: {e}")
            return "Unknown", 0.0

    def predict_gender(self, face_img):
        try:
            blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227),
                                         [78.4263377603, 87.7689143744, 114.895847746],
                                         swapRB=False)
            self.gender_net.setInput(blob)
            preds = self.gender_net.forward()
            gender_idx = preds[0].argmax()
            return self.GENDER_CLASSES[gender_idx], float(preds[0][gender_idx])
        except Exception as e:
            print(f"Gender prediction failed: {e}")
            return "Unknown", 0.0

    def process_frame(self, frame):
        result_frame = frame.copy()
        faces = self.detect_faces(frame)
        results = []

        for i, (x, y, w, h, conf) in enumerate(faces):
            face_img = frame[y:y + h, x:x + w]
            if face_img.size == 0:
                print("Skipped empty face image")
                continue

            age, age_conf = self.predict_age(face_img)
            gender, gender_conf = self.predict_gender(face_img)

            results.append({
                "face_id": i + 1,
                "position": (x, y, w, h),
                "face_confidence": round(conf, 2),
                "age_range": age,
                "age_confidence": round(age_conf, 2),
                "gender": gender,
                "gender_confidence": round(gender_conf, 2)
            })

            #ui
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{gender} ({gender_conf:.2f}), Age: {age} ({age_conf:.2f})"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(result_frame, (x, y - 25), (x + label_size[0], y), (0, 255, 0), -1)
            cv2.putText(result_frame, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        #this display the fps
        if hasattr(self, 'fps'):
            cv2.putText(result_frame, f"FPS: {self.fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return result_frame, results

    def start_webcam(self):
        print("Starting webcam... Press 'q' to quit")
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open webcam")
            return

        self.fps = 0
        frame_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 1.0:
                self.fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()

            result_frame, _ = self.process_frame(frame)
            cv2.imshow("Age and Gender Detection", result_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("Webcam closed")

if __name__ == "__main__":
    try:
        detector = AgeGenderDetector()
        detector.start_webcam()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
    except Exception as e:
        print(f"\nError: {e}")
        
        