import threading
import cv2

from deepface import DeepFace

# 0 here is default camera - can experiment with others if you have them
cap = cv2.VideoCapture(0)

# Set the capture window size
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0

face_match = False

model = DeepFace.build_model("Facenet")
reference_img = cv2.imread("reference.jpg")

print("Reference Image Loaded.")

def check_face(frame):
    global face_match
    try:
        # Preprocess the frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Perform face verification
        result = DeepFace.verify(frame,reference_img,model_name="Facenet")
        face_match = result['verified']
    except ValueError:
        face_match = False

while True:
    # read() returns TWO values - the first tells us if the 2nd has real data or is empty
    ret, frame = cap.read()

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        break

    if ret:
        if counter % 30 == 0:
            try:
                print("Creating Thread to verify...")
                #note , after frame.copy() to make a Tuple
                threading.Thread(target=check_face, args=(frame.copy(),)).start()
            except ValueError:
                pass
        counter += 1

        if face_match:
            cv2.putText(frame, "MATCH!", (20,450), cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0), 3)
        else:
            cv2.putText(frame, "NO MATCH!", (20,450), cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255), 3)
            
        cv2.imshow("Video", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cv2.destroyAllWindows()