import tensorflow as tf
import numpy as np
import os
import cv2
import face_recognition
import align.detect_face


FPS = 30

DB_ROOT_DIR = "./Face_Database"

detector = cv2.CascadeClassifier('haar_alt.xml')

# Create MTCNN
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    sess_ = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess_.as_default():
        pnet, rnet, onet = align.detect_face.create_mtcnn(sess_, None)


def detect_face(img, face_size=1, detector_id=0):
    """
    Detect face in image, 3 detector are available.
    :param img: Image of user
    :param face_size: Minimum detected face size
    :param detector_id: ID of face detector
    :return:
    """
    if detector_id == 0:
        # MTCNN
        face_locations = []
        bounding_boxes, _ = align.detect_face.detect_face(img=img,
                                                          minsize=20 * face_size,
                                                          pnet=pnet, rnet=rnet, onet=onet,
                                                          threshold=[0.6, 0.7, 0.7],
                                                          factor=0.709)
        for i in range(bounding_boxes.shape[0]):  # Convert the face boxes to the format required by face_recognition
            face_locations.append((int(bounding_boxes[i, 1]),
                                   int(bounding_boxes[i, 2]),
                                   int(bounding_boxes[i, 3]),
                                   int(bounding_boxes[i, 0])))
    elif detector_id == 1:
        # OpenCV haar cascade classifier
        face_locations = []
        face_positions = detector.detectMultiScale(image=img,
                                                   scaleFactor=1.1,
                                                   minNeighbors=3,
                                                   minSize=(20 * face_size, 20 * face_size),
                                                   maxSize=(240, 240))
        for face_position in face_positions:  # Convert the face boxes to the format required by face_recognition
            face_locations.append((int(face_position[1]),
                                   int(face_position[0] + face_position[3]),
                                   int(face_position[1] + face_position[2]),
                                   int(face_position[0])))
    elif detector_id == 2:
        # HOG detector in face_recognition API
        face_locations = face_recognition.face_locations(img=img,
                                                         number_of_times_to_upsample=1,
                                                         model="hog",)
    else:
        print('Invalid Detector ID!')
        return

    return face_locations


def realtime_recognition(group_name, face_size=1, track_interval=200, recognition_interval=2000, scale_factor=1, tolerance=0.6):
    """
    Run realtime face recognition.
    :param group_name: Name of user group
    :param face_size: Minimum detected face size
    :param track_interval: Face detect interval/ms
    :param recognition_interval: Face recognize interval/ms
    :param scale_factor: Image processing zoom factor
    :param tolerance: Face recognition threshold
    :return: None
    """
    # Load face database
    db_path = os.path.join(DB_ROOT_DIR, group_name)

    known_face_encodings = []  # Features in database
    known_face_names = []  # Names in database

    for person in os.listdir(db_path):
        known_face_encodings.append(np.load(os.path.join(db_path, person)))
        known_face_names.append(person.replace(".", "_").split("_")[0])

    face_locations = []  # Container for detected face boxes
    face_names = []  # Container for recognized face names

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    ret, frame = cap.read()

    timer = 0  # Frame skip timer
    while ret:
        timer += 1
        ret, frame = cap.read()

        # Face detection
        if timer % (track_interval * FPS // 1000) == 0:
            small_frame = cv2.resize(frame, (0, 0), fx=1 / scale_factor, fy=1 / scale_factor)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            # gray_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            face_locations = detect_face(rgb_small_frame, face_size)

        # Face recognition
        if timer % (recognition_interval * FPS // 1000) == 0 and face_locations != []:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Encode faces into 128-dimensional features
            face_encodings = face_recognition.face_encodings(face_image=rgb_frame,
                                                             known_face_locations=face_locations * scale_factor,
                                                             num_jitters=1
                                                             )
            face_names.clear()
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings=known_face_encodings,
                                                         face_encoding_to_check=face_encoding,
                                                         tolerance=tolerance
                                                         )
                name = "Unknown"
                face_distances = face_recognition.face_distance(face_encodings=known_face_encodings,
                                                                face_to_compare=face_encoding
                                                                )

                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[int(best_match_index)]
                face_names.append(name)

        # Draw face boxes and names
        for (top, right, bottom, left), name in zip(face_locations, face_names):

            top *= scale_factor
            right *= scale_factor
            bottom *= scale_factor
            left *= scale_factor

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 1)
            cv2.rectangle(frame, (left, bottom), (right, int(bottom + (bottom - top) * 0.25)), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, name, (left, int(bottom + (bottom - top) * 0.24)),
                        cv2.FONT_HERSHEY_DUPLEX, (right - left) / 120, (255, 255, 255), 1)

        cv2.imshow('camera', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyWindow("camera")


if __name__ == '__main__':
    realtime_recognition(group_name="test_group", face_size=1, track_interval=200, recognition_interval=1000, scale_factor=1, tolerance=0.6)
