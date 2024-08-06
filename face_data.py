import cv2
import numpy as np

# Face detection and data collection function
def collect_face_data(cap, face_cascade, dataset_path, file_name):
    collected_face_data = []
    frame_skip_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

        if len(faces) == 0:
            continue

        faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)

        for face in faces[:1]:
            x, y, w, h = face
            offset = 5
            face_offset = frame[y-offset:y+h+offset, x-offset:x+w+offset]
            face_selection = cv2.resize(face_offset, (100, 100))

            if frame_skip_counter % 10 == 0:
                collected_face_data.append(face_selection)
                print(len(collected_face_data))

            cv2.namedWindow("Face Selection", cv2.WINDOW_NORMAL)
            cv2.imshow("Face Selection", face_selection)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Faces", frame)

        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == ord('q'):
            break

        frame_skip_counter += 1

    return collected_face_data

# Data processing and saving function
def process_and_save_face_data(face_data, dataset_path, file_name):
    face_data = np.array(face_data)
    face_data = face_data.reshape((face_data.shape[0], -1))
    print(face_data.shape)

    np.save(dataset_path + file_name, face_data)
    print("Dataset saved at: {}".format(dataset_path + file_name + '.npy'))

# Main loop and user input handling
def main():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")

    if face_cascade.empty():
        print("Failed to load face detection model")
        return

    dataset_path = "./face_dataset/"
    file_name = input("Enter the name of person: ")

    collected_face_data = collect_face_data(cap, face_cascade, dataset_path, file_name)
    process_and_save_face_data(collected_face_data, dataset_path, file_name)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()