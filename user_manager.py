import tensorflow as tf
import numpy as np
import align.detect_face
import face_recognition
import cv2
import os


DB_ROOT_DIR = "./Face_Database"

detector = cv2.CascadeClassifier('haar_alt.xml')

# Create MTCNN
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)


def detect_face(img, detector_id=0):
    """
    Detect face in image, 3 detector are available.
    :param img: Image of user
    :param detector_id: ID of face detector
    :return:
    """
    if detector_id == 0:
        # MTCNN
        face_locations = []
        bounding_boxes, _ = align.detect_face.detect_face(img=img,
                                                          minsize=20,
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
                                                   minNeighbors=3,)
        for face_position in face_positions:  # Convert the face boxes to the format required by face_recognition
            face_locations.append((int(face_position[1]),
                                   int(face_position[0] + face_position[3]),
                                   int(face_position[1] + face_position[2]),
                                   int(face_position[0])))
    elif detector_id == 2:
        # HOG detector in face_recognition API
        face_locations = face_recognition.face_locations(img=img,
                                                         number_of_times_to_upsample=1,
                                                         model="hog"
                                                         )
    else:
        print('Invalid Detector ID!')
        return

    return face_locations


def add_user(img_path, group_name):
    """
    Add a single user, save face in user image as feature vector.
    :param img_path: Path to user image
    :param group_name: Name of user group
    :return: None
    """
    db_path = os.path.join(DB_ROOT_DIR, group_name)
    if not os.path.exists(db_path):
        os.makedirs(db_path)

    if not os.path.exists(img_path):
        print("Invalid Image Path: " + img_path + "!")
        return

    extension = img_path.split(".")[-1]
    if extension != "jpg" and extension != "png" and extension != "jfif":
        print("Invalid Image Path: " + img_path + "!")
        return

    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    face_locations = detect_face(img)

    if len(face_locations) > 1:
        print("More than One Face in Image!")
        return
    if len(face_locations) < 1:
        print("No Faces in Image!")
        return

    # Encode faces into 128-dimensional features
    face_enc = face_recognition.face_encodings(face_image=img,
                                               known_face_locations=face_locations,
                                               num_jitters=1
                                               )[0]

    name = str(img_path.split(".")[-2].split("/")[-1])
    np.save(os.path.join(db_path, name), face_enc)
    print(name + " Has Been Added!")


def add_user_batch(img_path, group_name):
    """
    Add users in batches, read all images in the path and save faces as feature vectors
    :param img_path: Path to the folder holding user images
    :param group_name: Name of user group
    :return: None
    """
    db_path = os.path.join(DB_ROOT_DIR, group_name)
    if not os.path.exists(db_path):
        os.makedirs(db_path)

    if not os.path.exists(img_path):
        print("Invalid Image Path: " + img_path + "!")
        return

    for image in os.listdir(img_path):

        extension = image.split(".")[-1]
        if extension == "jpg" or extension == "png" or extension == "jfif":
            img = cv2.cvtColor(cv2.imread(os.path.join(img_path, image)), cv2.COLOR_BGR2RGB)

            face_locations = detect_face(img)

            if len(face_locations) > 1:
                print("More than One Face in Image: " + image + "!")
                continue
            if len(face_locations) < 1:
                print("No Faces in Image: " + image + "!")
                continue

            # Encode faces into 128-dimensional features
            face_enc = face_recognition.face_encodings(face_image=img,
                                                       known_face_locations=face_locations,
                                                       num_jitters=1
                                                       )[0]

            name = str(image.split(".")[0])
            np.save(os.path.join(db_path, name), face_enc)
            print(name + " Has Been Added!")

    print("User Add Finished!")


def delete_user(user_name, group_name):
    """
    Delete all data of specified user.
    :param user_name: Name of user
    :param group_name: Name of user group
    :return: None
    """
    db_path = os.path.join(DB_ROOT_DIR, group_name)
    if not os.path.exists(db_path):
        print("Group Not Exist: " + group_name + "!")
        return

    delete_flag = False
    for feature in os.listdir(db_path):
        if feature.replace(".", "_").split("_")[0] == user_name:
            delete_flag = True
            try:
                os.remove(os.path.join(db_path, feature))
                print(os.path.join(db_path, feature) + " Has Been Deleted!")
            except OSError:
                print(os.path.join(db_path, feature) + " Cannot be Deleted!")
    if not delete_flag:
        print("User Not Exist: " + user_name + "!")


def delete_user_group(group_name):
    """
    Delete all data in specified user group.
    :param group_name: Name of user group
    :return: None
    """
    db_path = os.path.join(DB_ROOT_DIR, group_name)

    exist = os.path.exists(db_path)
    if not exist:
        print("Group Not Exist: " + group_name + "!")
        return

    for feature in os.listdir(db_path):
        try:
            os.remove(os.path.join(db_path, feature))
        except OSError:
            print(db_path + feature + " Cannot be Deleted!")

    try:
        os.rmdir(db_path)
        print(group_name + " Has Been Deleted!")
    except OSError:
        print(db_path + " Cannot be Deleted!")


if __name__ == '__main__':
    # add_user(img_path="./Face_Image/Hu Ge.jfif", group_name="test_group")
    add_user_batch(img_path="./Face_Image", group_name="test_group")
    # delete_user(user_name="HuGe", group_name="test_group")
    # delete_user_group(group_name="test_group")
