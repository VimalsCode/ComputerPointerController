import cv2

"""
Helper utility used to visualize the predicted output from the model.
"""


def draw_head_pose_estimation(frame, head_pose_estimation_output):
    """
    To draw the output from head pose estimation model
    :param frame:input frame
    :param head_pose_estimation_output:output
    :return:frame with the text information
    """
    frame_text = "head pose: (y={:.2f}, p={:.2f}, r={:.2f})".format(head_pose_estimation_output[0],
                                                                    head_pose_estimation_output[1],
                                                                    head_pose_estimation_output[2])
    return cv2.putText(frame, frame_text, (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.35, (255, 255, 255), 1)


def draw_face_detection(frame, detected_box):
    """
    To draw the output from face detection model
    :param frame: input frame
    :param detected_box: detected output
    :return: frame with face detection
    """
    return cv2.rectangle(frame, (detected_box[0], detected_box[1]), (detected_box[2], detected_box[3]), (0, 55, 255),
                         1)


def draw_facial_landmarks_estimation(detected_face_image, left_eye_dimension, right_eye_dimension):
    """
    Reference : https://knowledge.udacity.com/questions/245775
    :param detected_face_image:frame
    :param left_eye_dimension:left eye dimension
    :param right_eye_dimension:right eye dimension
    :return:frame with eye dimension drawn
    """
    # draw bounding box on left eye
    cv2.rectangle(detected_face_image, (left_eye_dimension[0], left_eye_dimension[1]),
                  (left_eye_dimension[2], left_eye_dimension[3]),
                  (0, 55, 255), 1)

    # draw bounding box on right eye
    cv2.rectangle(detected_face_image, (right_eye_dimension[0], right_eye_dimension[1]),
                  (right_eye_dimension[2], right_eye_dimension[3]),
                  (0, 55, 255), 1)
    return detected_face_image


def draw_gaze_estimation(eye_bounding_box, predicted_gaze_output, frame):
    """
    To draw gaze estimation output on the frame
    Reference : https://knowledge.udacity.com/questions/257811
    :param eye_bounding_box: face landmark detection output
    :param predicted_gaze_output: gaze vector
    :param frame: image frame
    :return:
    """
    left_eye_center1 = int(eye_bounding_box[0] + predicted_gaze_output[0] * 100)
    left_eye_center2 = int(eye_bounding_box[1] - predicted_gaze_output[1] * 100)
    draw_arrow(frame, (eye_bounding_box[0], eye_bounding_box[1]), (left_eye_center1, left_eye_center2))

    right_eye_center1 = int(eye_bounding_box[2] + predicted_gaze_output[0] * 100)
    right_eye_center2 = int(eye_bounding_box[3] - predicted_gaze_output[1] * 100)
    draw_arrow(frame, (eye_bounding_box[2], eye_bounding_box[3]), (right_eye_center1, right_eye_center2))
    return frame


def draw_arrow(frame, point1, point2):
    """
    To draw the arrow line
    :param frame: image frame
    :param point1: bounding box1
    :param point2: bounding box1
    :return: frame with arrowedLine drawn
    """

    return cv2.arrowedLine(frame, point1, point2, (255, 0, 0), 2)
