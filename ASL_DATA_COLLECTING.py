import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "holistic_landmarker.task"

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HolisticLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    min_pose_detection_confidence=0.5,
    min_pose_landmarks_confidence=0.5,
    min_hand_landmarks_confidence=0.5,
)

cap = cv2.VideoCapture(0)
timestamp_ms = 0

with vision.HolisticLandmarker.create_from_options(options) as landmarker:
  while cap.isOpened():
    success, frame = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    timestamp_ms += 33  
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    height, width, _ = frame.shape

    if result.pose_landmarks:
      right_shoulder = result.pose_landmarks[12]

    if result.right_hand_landmarks:
      wrist = result.right_hand_landmarks[0]
      if result.pose_landmarks:
        anchored_x = wrist.x - right_shoulder.x
        anchored_y = wrist.y - right_shoulder.y
        print(
            f"\rWrist relative to shoulder -> X: {anchored_x:.3f} | Y:"
            f" {anchored_y:.3f}      ",
            end="",
            flush=True,
        )

    cv2.imshow("HOLISTIC LANDMARKS", frame)

    if cv2.waitKey(5) & 0xFF == ord("q"):
      break

cap.release()
cv2.destroyAllWindows()