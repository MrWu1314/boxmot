# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license
import json
import os
import argparse
import cv2
import numpy as np
from functools import partial
from pathlib import Path
import time
import torch
import tensorflow as tf
from boxmot import TRACKERS
from boxmot.tracker_zoo import create_tracker
from boxmot.utils import ROOT, WEIGHTS, TRACKER_CONFIGS
from boxmot.utils.checks import RequirementsChecker
from tracking.detectors import get_yolo_inferer

checker = RequirementsChecker()
checker.check_packages(('ultralytics @ git+https://github.com/mikel-brostrom/ultralytics.git',))  # install

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.data.utils import VID_FORMATS
from ultralytics.utils.plotting import save_one_box
# from retinaface import RetinaFace
from deepface import DeepFace


def calculate_face_angles(landmarks):
    # 提取关键点坐标
    left_eye = np.array(landmarks['left_eye'])
    right_eye = np.array(landmarks['right_eye'])
    nose = np.array(landmarks['nose'])
    mouth_left = np.array(landmarks['mouth_left'])
    mouth_right = np.array(landmarks['mouth_right'])

    # 计算眼睛间距
    dPx_eyes = right_eye[0] - left_eye[0]
    if int(dPx_eyes) == 0:
        dPx_eyes = 1

    #dPx_eyes = max((right_eye[0] - left_eye[0]), 1)  # 保证不为0
    dPy_eyes = right_eye[1] - left_eye[1]

    # 计算旋转角度
    angle = np.arctan(dPy_eyes / dPx_eyes)

    # 计算旋转矩阵的 cos 和 sin
    alpha = np.cos(angle)
    beta = np.sin(angle)

    # 旋转后的坐标
    LMx = np.array([left_eye[0], right_eye[0], nose[0], mouth_left[0], mouth_right[0]])
    LMy = np.array([left_eye[1], right_eye[1], nose[1], mouth_left[1], mouth_right[1]])

    # 旋转后的坐标
    LMxr = (alpha * LMx + beta * LMy + (1 - alpha) * LMx[2] / 2 - beta * LMy[2] / 2)
    LMyr = (-beta * LMx + alpha * LMy + beta * LMx[2] / 2 + (1 - alpha) * LMy[2] / 2)

    # 计算眼睛和嘴巴之间的平均距离
    dXtot = (LMxr[1] - LMxr[0] + LMxr[4] - LMxr[3]) / 2
    dYtot = (LMyr[3] - LMyr[0] + LMyr[4] - LMyr[1]) / 2

    # 计算鼻子和眼睛之间的平均距离
    dXnose = (LMxr[1] - LMxr[2] + LMxr[4] - LMxr[2]) / 2
    dYnose = (LMyr[3] - LMyr[2] + LMyr[4] - LMyr[2]) / 2

    # 计算前脸角度，0度表示正面，90度表示侧面
    Xfrontal = (-90 + 90 / 0.5 * dXnose / dXtot) if dXtot != 0 else 0
    Yfrontal = (-90 + 90 / 0.5 * dYnose / dYtot) if dYtot != 0 else 0

    # 返回旋转角度和前脸角度
    # return angle * 180 / np.pi, Xfrontal, Yfrontal
    roll = angle * 180 / np.pi
    yaw = Xfrontal
    pitch = Yfrontal

    pitch_angle = max(min(pitch, 89), -89)
    yaw_angle = max(min(yaw, 89), -89)
    roll_angle = max(min(roll, 89), -89)
    print("all degrees is :", abs(pitch_angle * 0.1) + abs(yaw_angle * 0.7) + abs(
        roll_angle * 0.2))  #abs(pitch * 0.1) + abs(roll * 0.2) + abs(yaw * 0.7):

    return pitch_angle, yaw_angle, roll_angle

def detect_faces_angles(image):
    """
    使用 RetinaFace 模型检测图像中的人脸
    :param image: 输入图像
    :return: 检测到的面部信息
    """
    pitch = 90.0
    yaw = 90.0
    roll = 90.0
    age = -1
    start_time = time.time()
    try:
        # 使用 RetinaFace 检测人脸
        face_objs = DeepFace.analyze(image, detector_backend="retinaface",
                                     actions=['age'],
                                     anti_spoofing=False,
                                     enforce_detection=False)
        face = face_objs[0]
        age = face['age']
        face_area = face['region']
        face_landmarks = {'left_eye': [face_area['left_eye'][0], face_area['left_eye'][1]],
                          'right_eye': [face_area['right_eye'][0], face_area['right_eye'][1]],
                          'nose': [face_area['nose'][0], face_area['nose'][1]],
                          'mouth_left': [face_area['mouth_left'][0], face_area['mouth_left'][1]],
                          'mouth_right': [face_area['mouth_right'][0], face_area['mouth_right'][1]]}
        pitch, yaw, roll = calculate_face_angles(face_landmarks)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"image:{image.shape} detect_faces_angles函数执行的时间: {execution_time} 秒")
        return pitch, yaw, roll, age
    except Exception as e:
        print(f"检测失败: {e}")
        return pitch, yaw, roll, age


import cv2


def get_timestamp_by_frame(video_path, frame_num):
    """
    根据帧序号获取视频时间戳
    :param video_path: 视频文件路径
    :param frame_num: 目标帧序号(从0开始)
    :return: 时间戳(毫秒)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("无法打开视频文件")

    # 获取视频帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 设置目标帧位置
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    # 获取时间戳(毫秒)
    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)

    cap.release()
    return timestamp


# 使用示例
if __name__ == "__main__":
    video = "test.mp4"  # 替换为你的视频路径
    frame_index = 100  # 目标帧序号
    ms = get_timestamp_by_frame(video, frame_index)
    print(f"第{frame_index}帧对应时间戳: {ms:.0f}毫秒 ({ms / 1000:.2f}秒)")



def on_predict_start(predictor, persist=False):
    """
    Initialize trackers for object tracking during prediction.

    Args:
        predictor (object): The predictor object to initialize trackers for.
        persist (bool, optional): Whether to persist the trackers if they already exist. Defaults to False.
    """

    assert predictor.custom_args.tracking_method in TRACKERS, \
        f"'{predictor.custom_args.tracking_method}' is not supported. Supported ones are {TRACKERS}"

    tracking_config = TRACKER_CONFIGS / (predictor.custom_args.tracking_method + '.yaml')
    trackers = []
    for i in range(predictor.dataset.bs):
        tracker = create_tracker(
            predictor.custom_args.tracking_method,
            tracking_config,
            predictor.custom_args.reid_model,
            predictor.device,
            predictor.custom_args.half,
            predictor.custom_args.per_class
        )
        # motion only modeles do not have
        if hasattr(tracker, 'model'):
            tracker.model.warmup()
        trackers.append(tracker)

    predictor.trackers = trackers


@torch.no_grad()
def run(args):
    print("args:",args)
    vid_writer = None
    target_save_path = f"{args.custom_save_path}/target_video"
    if args.custom_save_path is not None:
        if not os.path.exists(args.custom_save_path):
            os.makedirs(args.custom_save_path)

        if not os.path.exists(target_save_path):
            os.makedirs(target_save_path)
    face_angle_threshold = max(min(args.max_face_angle, 89), 0)
    min_age = max(min(100, args.min_age), 0)
    max_age = max(min(100, args.max_age), 0)
    args.min_age = min(min_age, max_age)
    args.max_age = max(min_age, max_age)
    age_threshold = (args.min_age, args.max_age)

    ul_models = ['yolov8', 'yolov9', 'yolov10', 'yolo11', 'rtdetr', 'sam']

    yolo = YOLO(
        args.yolo_model if any(yolo in str(args.yolo_model) for yolo in ul_models) else 'yolov8n.pt',
    )

    results = yolo.track(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        agnostic_nms=args.agnostic_nms,
        show=False,
        stream=True,
        device=args.device,
        show_conf=args.show_conf,
        save_txt=args.save_txt,
        show_labels=args.show_labels,
        save=args.save,
        verbose=args.verbose,
        exist_ok=args.exist_ok,
        project=args.project,
        name=args.name,
        classes=args.classes,
        imgsz=args.imgsz,
        vid_stride=args.vid_stride,
        line_width=args.line_width
    )

    yolo.add_callback('on_predict_start', partial(on_predict_start, persist=True))

    if not any(yolo in str(args.yolo_model) for yolo in ul_models):
        # replace yolov8 model
        m = get_yolo_inferer(args.yolo_model)
        model = m(
            model=args.yolo_model,
            device=yolo.predictor.device,
            args=yolo.predictor.args
        )
        yolo.predictor.model = model

    # store custom args in predictor
    yolo.predictor.custom_args = args
    frames = 0
    for r in results:
        if frames == 0 or frames % 5 == 0:
            yolo.predictor.trackers[0].save_target_video(r.orig_img, target_save_path,frames=frames,
                                                         face_detect_func=detect_faces_angles)
        else:
            yolo.predictor.trackers[0].save_target_video(r.orig_img, target_save_path,frames=frames)
        img = yolo.predictor.trackers[0].plot_results(r.orig_img, args.show_trajectories)

        if args.show is True:
            cv2.imshow('BoxMOT', img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') or key == ord('q'):
                break
        if args.custom_save_path is not None:
            if vid_writer is None:
                width = img.shape[1]
                height = img.shape[0]
                target_video_path = f"{args.custom_save_path}/custom_video.mp4"
                vid_writer = cv2.VideoWriter(
                    target_video_path, cv2.VideoWriter_fourcc(*"mp4v"), 25, (int(width), int(height))
                )
            vid_writer.write(img)
        frames += 1
    yolo.predictor.trackers[0].check_target(face_angle_threshold=face_angle_threshold, age_threshold=age_threshold)
    data = yolo.predictor.trackers[0].out_target_json(args.source)
    # print("test pso json indent2:")
    with open(f"{args.custom_save_path}/result.json", 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    # data=json.dumps(data, indent=2, ensure_ascii=False)
    # print(data)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-model', type=Path, default=WEIGHTS / 'yolo11x.pt',
                        help='yolo model path')
    parser.add_argument('--reid-model', type=Path, default=WEIGHTS / 'osnet_x1_0_msmt17.pt',
                        help='reid model path')
    parser.add_argument('--tracking-method', type=str, default='botsort',
                        help='deepocsort, botsort, strongsort, ocsort, bytetrack, imprassoc')
    parser.add_argument('--source', type=str, default='0',
                        help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[720, 1280],
                        help='inference size h,w')
    parser.add_argument('--conf', type=float, default=0.1,
                        help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7,
                        help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--device', default='0',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show', action='store_true',
                        help='display tracking video results')
    parser.add_argument('--save', action='store_true',
                        help='save video tracking results')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int,default=0,
                        help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--project', default=ROOT / 'runs' / 'track',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true',
                        help='use FP16 half-precision inference')
    parser.add_argument('--vid-stride', type=int, default=1,
                        help='video frame-rate stride')
    parser.add_argument('--show-labels', action='store_false',
                        help='either show all or only bboxes')
    parser.add_argument('--show-conf', action='store_false',
                        help='hide confidences when show')
    parser.add_argument('--show-trajectories', action='store_true',
                        help='show confidences')
    parser.add_argument('--save-txt', action='store_true',
                        help='save tracking results in a txt file')
    parser.add_argument('--save-id-crops', action='store_true',
                        help='save each crop to its respective id folder')
    parser.add_argument('--line-width', default=None, type=int,
                        help='The line width of the bounding boxes. If None, it is scaled to the image size.')
    parser.add_argument('--per-class', default=False, action='store_true',
                        help='not mix up classes when tracking')
    parser.add_argument('--verbose', default=True, action='store_true',
                        help='print results per frame')
    parser.add_argument('--agnostic-nms', default=False, action='store_true',
                        help='class-agnostic NMS')

    parser.add_argument('--custom-save-path', type=str, default=ROOT / 'runs' / 'custom_track',
                        help='custom save video tracking results')

    parser.add_argument('--custom-videos', type=bool, default=True,
                        help='custom save video tracking results')

    parser.add_argument('--min-age', type=int, default=1, help='target min age')
    parser.add_argument('--max-age', type=int, default=100, help='target max age')
    parser.add_argument('--max-face-angle', type=int, default=89, help='target max face angle(pitch,yaw,roll)')

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    opt = parse_opt()
    run(opt)
