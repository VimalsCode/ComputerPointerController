import argparse
import logging as log
import os
import statistics
import time

import cv2
from face_detection import FaceDetectionModel
from facial_landmarks_detection import FacialLandmarksDetectionModel
from gaze_estimation import GazeEstimationModel
from head_pose_estimation import HeadPoseEstimationModel
from input_feeder import InputFeeder
from mouse_controller import MouseController

"""
Project Main file to load the model and perform inference.
"""


def analyze_model_inference_time(m_fd_infer_time, m_hpe_infer_time, m_fld_infer_time, m_ge_infer_time, fp_type="FP32"):
    """
    To generate model analysis
    :param m_fd_infer_time: face detection inference time
    :param m_hpe_infer_time: head pose estimation inference time
    :param m_fld_infer_time: face landmark detection inference time
    :param m_ge_infer_time: gaze estimation inference time
    :param fp_type: precision type to use
    :return: None
    """

    if fp_type == "FP32":
        file_name = "analysis/model_inference_fp32.txt"
    else:
        file_name = "analysis/model_inference_fp16.txt"
    with open(file_name, "w") as f:
        m_fd_avg_inf = round(statistics.mean(m_fd_infer_time), 5)
        m_hpe_avg_inf = round(statistics.mean(m_hpe_infer_time), 5)
        m_fld_avg_inf = round(statistics.mean(m_fld_infer_time), 5)
        m_ge_avg_inf = round(statistics.mean(m_ge_infer_time), 5)
        f.write(str(m_fd_avg_inf) + '\n')
        f.write(str(m_hpe_avg_inf) + '\n')
        f.write(str(m_fld_avg_inf) + '\n')
        f.write(str(m_ge_avg_inf) + '\n')

    inference_time = []
    f = open(file_name, 'r')
    for content in f:
        inference_time.append(float(content))


def main():
    # create log file
    log.basicConfig(filename='logs/cpc.log', level=log.INFO, format='%(asctime)s %(message)s')

    # Parse the argument
    args = parse_arguments().parse_args()

    print('Input arguments:')
    for key, value in vars(args).items():
        print('\t{}: {}'.format(key, value))
    print('')

    # list used to handle model load and inference time
    m_fd_load_time = []
    m_hpe_load_time = []
    m_fd_infer_time = []
    m_hpe_infer_time = []
    m_fld_infer_time = []
    m_ge_infer_time = []
    m_fld_load_time = []
    m_ge_load_time = []

    if args.input == 'CAM':
        input_feeder = InputFeeder("cam")
    else:
        # get the input value
        input_stream = args.input
        if not os.path.isfile(input_stream):
            log.error("Provided input video file doesn't exist/video path is wrong!")
            exit(1)
        # load the video file
        input_feeder = InputFeeder("video", input_stream)

    # get model path
    head_face_detection_model_name = args.face_detection
    head_pose_estimation_model_name = args.head_pose_estimation
    face_landmarks_detection_model_name = args.facial_landmarks_detection
    gaze_estimation_model_name = args.gaze_estimation

    # mouse controller
    mouse_controller = MouseController(precision='medium', speed='fast')

    # load the required models
    m_fd_load_start_time = time.time()
    # create and load face_detection model
    face_detection = FaceDetectionModel(model_name=head_face_detection_model_name, device=args.device,
                                        probs_threshold=args.prob_threshold)
    face_detection.load_model()
    m_fd_load_time.append(round(time.time() - m_fd_load_start_time, 5))
    log.debug("Time taken to load Face detection model took {} seconds.".format(m_fd_load_time))

    # create and load head_pose estimation model
    m_hpe_load_start_time = time.time()
    head_pose_estimation = HeadPoseEstimationModel(model_name=head_pose_estimation_model_name, device=args.device)
    head_pose_estimation.load_model()
    m_hpe_load_time.append(round(time.time() - m_hpe_load_start_time, 5))
    log.debug("Time taken to load head pose estimation model took {} seconds.".format(m_hpe_load_time))

    # create and load face landmarks detection model
    m_fld_load_start_time = time.time()
    face_landmark_detection = FacialLandmarksDetectionModel(model_name=face_landmarks_detection_model_name,
                                                            device=args.device)
    face_landmark_detection.load_model()
    m_fld_load_time.append(round(time.time() - m_fld_load_start_time, 5))
    log.debug("Time taken to load face landmark detection model took {} seconds.".format(m_fld_load_time))

    # create and load face landmarks detection model
    m_ge_load_start_time = time.time()
    gaze_estimation = GazeEstimationModel(model_name=gaze_estimation_model_name, device=args.device)
    gaze_estimation.load_model()
    m_ge_load_time.append(round(time.time() - m_ge_load_start_time, 5))
    log.debug("Time taken to load gaze estimation model took {} seconds.".format(m_ge_load_time))

    # load the image data
    input_feeder.load_data()
    frame_count = 0
    threshold_frame = 5

    log.info("Video stream to perform gaze estimation is started!.")
    for flag, frame in input_feeder.next_batch():
        if not flag:
            break
        # to handle better control with frame processing
        if frame_count % threshold_frame == 0:
            key_pressed = cv2.waitKey(60)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if key_pressed == 27:
                break
            # invoke face detection prediction
            m_fd_infer_start_time = time.time()
            detected_face_image, detected_box = face_detection.predict_face_detection(frame, args.visualization_fd)
            m_fd_infer_end_time = time.time()
            m_fd_infer_time.append(m_fd_infer_end_time - m_fd_infer_start_time)

            # invoke head pose estimation prediction
            head_pose_estimation_output, frame = head_pose_estimation.predict_head_pose_estimation(frame,
                                                                                                   detected_face_image,
                                                                                                   args.visualization_hpe)
            m_hpe_infer_end_time = time.time()
            m_hpe_infer_time.append(m_hpe_infer_end_time - m_fd_infer_end_time)

            # invoke face landmark detection prediction
            left_eye_image, right_eye_image, = face_landmark_detection.predict_facial_landmarks_detection(
                detected_face_image, args.visualization_fld)
            m_fld_infer_end_time = time.time()
            m_fld_infer_time.append(m_fld_infer_end_time - m_hpe_infer_end_time)

            # invoke gaze estimation prediction
            mouse_coordinate, predicted_gaze_output = gaze_estimation.predict_gaze_estimation(left_eye_image,
                                                                                              right_eye_image,
                                                                                              head_pose_estimation_output)
            m_ge_infer_end_time = time.time()
            m_ge_infer_time.append(m_ge_infer_end_time - m_fld_infer_end_time)

            if args.visualization_ge:
                # get the output from face landmark detection
                outputs = face_landmark_detection.get_outputs()
                # get back the bounding box
                height = detected_face_image.shape[0]
                width = detected_face_image.shape[1]
                left_eye_x = int(outputs[0] * width + detected_box[0])
                left_eye_y = int(outputs[1] * height + detected_box[1])
                right_eye_x = int(outputs[2] * width + detected_box[0])
                right_eye_y = int(outputs[3] * height + detected_box[1])
                eye_bounding_box = [left_eye_x, left_eye_y, right_eye_x, right_eye_y]
                gaze_estimation.draw_gaze_estimation(eye_bounding_box, predicted_gaze_output, frame)

            # show the results
            cv2.imshow('ComputerPointer', frame)
            mouse_controller.move(mouse_coordinate[0], mouse_coordinate[1])
        frame_count = frame_count + 1

    log.info("Completed gaze estimation for the provided video!.")
    log.info("Mean time taken to run Face detection inference took {} seconds.".format(statistics.mean(m_fd_infer_time)))
    log.info(
        "Mean time taken to run Head pose estimation inference took {} seconds.".format(statistics.mean(m_hpe_infer_time)))
    log.info("Mean time taken to run Face Landmark detection inference took {} seconds.".format(
        statistics.mean(m_fld_infer_time)))
    log.info("Mean time taken to run Gaze estimation inference took {} seconds.".format(statistics.mean(m_ge_infer_time)))
    # to perform model inference analysis
    # analyze_model_inference_time(m_fd_infer_time, m_hpe_infer_time, m_fld_infer_time, m_ge_infer_time, "FP32")
    # clean up resources
    input_feeder.close()
    cv2.destroyAllWindows()


def parse_arguments():
    argument_parser = argparse.ArgumentParser(description='Computer Pointer Controller implementation')

    argument_parser.add_argument("-d", "--device", type=str, default="CPU",
                                 help="Specify the target device to infer on: "
                                      "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                                      "will look for a suitable plugin for device "
                                      "specified (CPU by default)")

    argument_parser.add_argument("-i", "--input", required=True, type=str,
                                 help="Path to image or video file")

    argument_parser.add_argument("-fd", "--face_detection", required=True, type=str,
                                 help="Path to an xml file with a trained model.")

    argument_parser.add_argument("-fld", "--facial_landmarks_detection", required=True, type=str,
                                 help="Path to an xml file with a trained model.")

    argument_parser.add_argument("-ge", "--gaze_estimation", required=True, type=str,
                                 help="Path to an xml file with a trained model.")

    argument_parser.add_argument("-hpe", "--head_pose_estimation", required=True, type=str,
                                 help="Path to an xml file with a trained model.")

    # optional arguments
    argument_parser.add_argument("-pt", "--prob_threshold", type=float, required=False, default=0.5,
                                 help="Probability threshold for detections filtering"
                                      "(0.5 by default)")
    argument_parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                                 default=None,
                                 help="MKLDNN (CPU)-targeted custom layers."
                                      "Absolute path to a shared library with the"
                                      "kernels impl.")

    argument_parser.add_argument("-v_fd", "--visualization_fd", type=lambda x: x == 'True', default=True,
                                 required=False, help='Do not generate face detection visualization if False')
    argument_parser.add_argument("-v_hpe", "--visualization_hpe", type=lambda x: x == 'True', default=True,
                                 required=False, help='Do not generate head pose estimation visualization if False')
    argument_parser.add_argument("-v_fld", "--visualization_fld", type=lambda x: x == 'True', default=True,
                                 required=False,
                                 help='Do not generate facial landmark detection visualization if False')
    argument_parser.add_argument("-v_ge", "--visualization_ge", type=lambda x: x == 'True', default=True,
                                 required=False, help='Do not generate gaze estimation visualization if False')
    return argument_parser


if __name__ == '__main__':
    main()
