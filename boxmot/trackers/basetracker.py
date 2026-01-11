import os

import cv2
import numpy as np
import cv2 as cv
import hashlib
import colorsys
from abc import ABC, abstractmethod
from boxmot.utils import logger as LOGGER
from boxmot.utils.iou import AssociationFunction
import subprocess
import json

class BaseTracker(ABC):
    def __init__(
        self, 
        det_thresh: float = 0.3,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        max_obs: int = 50,
        nr_classes: int = 80,
        per_class: bool = False,
        asso_func: str = 'iou'
    ):
        """
        Initialize the BaseTracker object with detection threshold, maximum age, minimum hits, 
        and Intersection Over Union (IOU) threshold for tracking objects in video frames.

        Parameters:
        - det_thresh (float): Detection threshold for considering detections.
        - max_age (int): Maximum age of a track before it is considered lost.
        - min_hits (int): Minimum number of detection hits before a track is considered confirmed.
        - iou_threshold (float): IOU threshold for determining match between detection and tracks.

        Attributes:
        - frame_count (int): Counter for the frames processed.
        - active_tracks (list): List to hold active tracks, may be used differently in subclasses.
        """
        self.det_thresh = det_thresh
        self.max_age = max_age
        self.max_obs = max_obs
        self.min_hits = min_hits
        self.per_class = per_class  # Track per class or not
        self.nr_classes = nr_classes
        self.iou_threshold = iou_threshold
        self.last_emb_size = None
        self.asso_func_name = asso_func

        self.frame_count = 0
        self.active_tracks = []  # This might be handled differently in derived classes
        self.per_class_active_tracks = None
        self._first_frame_processed = False  # Flag to track if the first frame has been processed
        
        # Initialize per-class active tracks
        if self.per_class:
            self.per_class_active_tracks = {}
            for i in range(self.nr_classes):
                self.per_class_active_tracks[i] = []
        
        if self.max_age >= self.max_obs:
            LOGGER.warning("Max age > max observations, increasing size of max observations...")
            self.max_obs = self.max_age + 5
            print("self.max_obs", self.max_obs)
        self.target_list = []

    @abstractmethod
    def update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None) -> np.ndarray:
        """
        Abstract method to update the tracker with new detections for a new frame. This method 
        should be implemented by subclasses.

        Parameters:
        - dets (np.ndarray): Array of detections for the current frame.
        - img (np.ndarray): The current frame as an image array.
        - embs (np.ndarray, optional): Embeddings associated with the detections, if any.

        Raises:
        - NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("The update method needs to be implemented by the subclass.")
    
    def get_class_dets_n_embs(self, dets, embs, cls_id):
        # Initialize empty arrays for detections and embeddings
        class_dets = np.empty((0, 6))
        class_embs = np.empty((0, self.last_emb_size)) if self.last_emb_size is not None else None

        # Check if there are detections
        if dets.size > 0:
            class_indices = np.where(dets[:, 5] == cls_id)[0]
            class_dets = dets[class_indices]
            
            if embs is not None:
                # Assert that if embeddings are provided, they have the same number of elements as detections
                assert dets.shape[0] == embs.shape[0], "Detections and embeddings must have the same number of elements when both are provided"
                
                if embs.size > 0:
                    class_embs = embs[class_indices]
                    self.last_emb_size = class_embs.shape[1]  # Update the last known embedding size
                else:
                    class_embs = None
        return class_dets, class_embs
    
    @staticmethod
    def on_first_frame_setup(method):
        """
        Decorator to perform setup on the first frame only.
        This ensures that initialization tasks (like setting the association function) only
        happen once, on the first frame, and are skipped on subsequent frames.
        """
        def wrapper(self, *args, **kwargs):
            # If setup hasn't been done yet, perform it
            if not self._first_frame_processed:
                img = args[1]
                self.h, self.w = img.shape[0:2]
                self.asso_func = AssociationFunction(w=self.w, h=self.h, asso_mode=self.asso_func_name).asso_func

                # Mark that the first frame setup has been done
                self._first_frame_processed = True

            # Call the original method (e.g., update)
            return method(self, *args, **kwargs)
        
        return wrapper
    
    
    @staticmethod
    def per_class_decorator(update_method):
        """
        Decorator for the update method to handle per-class processing.
        """
        def wrapper(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None):
            
            #handle different types of inputs
            if dets is None or len(dets) == 0:
                dets = np.empty((0, 6))
            
            if self.per_class:
                # Initialize an array to store the tracks for each class
                per_class_tracks = []
                
                # same frame count for all classes
                frame_count = self.frame_count

                for cls_id in range(self.nr_classes):
                    # Get detections and embeddings for the current class
                    class_dets, class_embs = self.get_class_dets_n_embs(dets, embs, cls_id)
                    
                    LOGGER.debug(f"Processing class {int(cls_id)}: {class_dets.shape} with embeddings {class_embs.shape if class_embs is not None else None}")

                    # Activate the specific active tracks for this class id
                    self.active_tracks = self.per_class_active_tracks[cls_id]
                    
                    # Reset frame count for every class
                    self.frame_count = frame_count
                    
                    # Update detections using the decorated method
                    tracks = update_method(self, dets=class_dets, img=img, embs=class_embs)

                    # Save the updated active tracks
                    self.per_class_active_tracks[cls_id] = self.active_tracks

                    if tracks.size > 0:
                        per_class_tracks.append(tracks)
                
                # Increase frame count by 1
                self.frame_count = frame_count + 1

                return np.vstack(per_class_tracks) if per_class_tracks else np.empty((0, 8))
            else:
                # Process all detections at once if per_class is False
                return update_method(self, dets=dets, img=img, embs=embs)
        return wrapper


    def check_inputs(self, dets, img):
        assert isinstance(
            dets, np.ndarray
        ), f"Unsupported 'dets' input format '{type(dets)}', valid format is np.ndarray"
        assert isinstance(
            img, np.ndarray
        ), f"Unsupported 'img_numpy' input format '{type(img)}', valid format is np.ndarray"
        assert (
            len(dets.shape) == 2
        ), "Unsupported 'dets' dimensions, valid number of dimensions is two"
        assert (
            dets.shape[1] == 6
        ), "Unsupported 'dets' 2nd dimension lenght, valid lenghts is 6"

    def id_to_color(self, id: int, saturation: float = 0.75, value: float = 0.95) -> tuple:
        """
        Generates a consistent unique BGR color for a given ID using hashing.

        Parameters:
        - id (int): Unique identifier for which to generate a color.
        - saturation (float): Saturation value for the color in HSV space.
        - value (float): Value (brightness) for the color in HSV space.

        Returns:
        - tuple: A tuple representing the BGR color.
        """

        # Hash the ID to get a consistent unique value
        hash_object = hashlib.sha256(str(id).encode())
        hash_digest = hash_object.hexdigest()
        
        # Convert the first few characters of the hash to an integer
        # and map it to a value between 0 and 1 for the hue
        hue = int(hash_digest[:8], 16) / 0xffffffff
        
        # Convert HSV to RGB
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        
        # Convert RGB from 0-1 range to 0-255 range and format as hexadecimal
        rgb_255 = tuple(int(component * 255) for component in rgb)
        hex_color = '#%02x%02x%02x' % rgb_255
        # Strip the '#' character and convert the string to RGB integers
        rgb = tuple(int(hex_color.strip('#')[i:i+2], 16) for i in (0, 2, 4))
        
        # Convert RGB to BGR for OpenCV
        bgr = rgb[::-1]
        
        return bgr

    def plot_box_on_img(self, img: np.ndarray, box: tuple, conf: float, cls: int, id: int, thickness: int = 2, fontscale: float = 0.5) -> np.ndarray:
        """
        Draws a bounding box with ID, confidence, and class information on an image.

        Parameters:
        - img (np.ndarray): The image array to draw on.
        - box (tuple): The bounding box coordinates as (x1, y1, x2, y2).
        - conf (float): Confidence score of the detection.
        - cls (int): Class ID of the detection.
        - id (int): Unique identifier for the detection.
        - thickness (int): The thickness of the bounding box.
        - fontscale (float): The font scale for the text.

        Returns:
        - np.ndarray: The image array with the bounding box drawn on it.
        """

        img = cv.rectangle(
            img,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            self.id_to_color(id),
            thickness
        )
        img = cv.putText(
            img,
            f'id: {int(id)}, conf: {conf:.2f}, c: {int(cls)}',
            (int(box[0]), int(box[1]) - 10),
            cv.FONT_HERSHEY_SIMPLEX,
            fontscale,
            self.id_to_color(id),
            thickness
        )
        return img

    def plot_box_on_img_copy(self, orig_img: np.ndarray, box: tuple, conf: float, cls: int, id: int, thickness: int = 2,
                             fontscale: float = 0.5) -> np.ndarray:
        """
        Draws a bounding box with ID, confidence, and class information on an image.

        Parameters:
        - img (np.ndarray): The image array to draw on.
        - box (tuple): The bounding box coordinates as (x1, y1, x2, y2).
        - conf (float): Confidence score of the detection.
        - cls (int): Class ID of the detection.
        - id (int): Unique identifier for the detection.
        - thickness (int): The thickness of the bounding box.
        - fontscale (float): The font scale for the text.

        Returns:
        - np.ndarray: The image array with the bounding box drawn on it.
        """
        img = np.ascontiguousarray(np.copy(orig_img))
        img = cv.rectangle(
            img,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            self.id_to_color(id),
            thickness
        )
        img = cv.putText(
            img,
            f'id: {int(id)}, conf: {conf:.2f}, c: {int(cls)}',
            (int(box[0]), int(box[1]) - 10),
            cv.FONT_HERSHEY_SIMPLEX,
            fontscale,
            self.id_to_color(id),
            thickness
        )
        return img

    def plot_trackers_trajectories(self, img: np.ndarray, observations: list, id: int) -> np.ndarray:
        """
        Draws the trajectories of tracked objects based on historical observations. Each point
        in the trajectory is represented by a circle, with the thickness increasing for more
        recent observations to visualize the path of movement.

        Parameters:
        - img (np.ndarray): The image array on which to draw the trajectories.
        - observations (list): A list of bounding box coordinates representing the historical
        observations of a tracked object. Each observation is in the format (x1, y1, x2, y2).
        - id (int): The unique identifier of the tracked object for color consistency in visualization.

        Returns:
        - np.ndarray: The image array with the trajectories drawn on it.
        """
        for i, box in enumerate(observations):
            trajectory_thickness = int(np.sqrt(float (i + 1)) * 1.2)
            img = cv.circle(
                img,
                (int((box[0] + box[2]) / 2),
                int((box[1] + box[3]) / 2)), 
                2,
                color=self.id_to_color(int(id)),
                thickness=trajectory_thickness
            )
        return img


    def plot_results(self, img: np.ndarray, show_trajectories: bool, thickness: int = 2, fontscale: float = 0.5) -> np.ndarray:
        """
        Visualizes the trajectories of all active tracks on the image. For each track,
        it draws the latest bounding box and the path of movement if the history of
        observations is longer than two. This helps in understanding the movement patterns
        of each tracked object.

        Parameters:
        - img (np.ndarray): The image array on which to draw the trajectories and bounding boxes.
        - show_trajectories (bool): Whether to show the trajectories.
        - thickness (int): The thickness of the bounding box.
        - fontscale (float): The font scale for the text.

        Returns:
        - np.ndarray: The image array with trajectories and bounding boxes of all active tracks.
        """

        # if values in dict
        if self.per_class_active_tracks is not None:
            for k in self.per_class_active_tracks.keys():
                active_tracks = self.per_class_active_tracks[k]
                for a in active_tracks:
                    if a.history_observations:
                        if len(a.history_observations) > 2:
                            box = a.history_observations[-1]
                            img = self.plot_box_on_img(img, box, a.conf, a.cls, a.id, thickness, fontscale)
                            if show_trajectories:
                                img = self.plot_trackers_trajectories(img, a.history_observations, a.id)
        else:
            for a in self.active_tracks:
                if a.history_observations:
                    if len(a.history_observations) > 2:
                        box = a.history_observations[-1]
                        img = self.plot_box_on_img(img, box, a.conf, a.cls, a.id, thickness, fontscale)
                        if show_trajectories:
                            img = self.plot_trackers_trajectories(img, a.history_observations, a.id)
                
        return img

    def save_target_video(self, orig_img: np.ndarray, custom_target_path: str, thickness: int = 2,
                          fontscale: float = 0.5, frames=None ,face_detect_func=None) -> np.ndarray:
        """
        Visualizes the trajectories of all active tracks on the image. For each track,
        it draws the latest bounding box and the path of movement if the history of
        observations is longer than two. This helps in understanding the movement patterns
        of each tracked object.

        Parameters:
        - img (np.ndarray): The image array on which to draw the trajectories and bounding boxes.
        - show_trajectories (bool): Whether to show the trajectories.
        - thickness (int): The thickness of the bounding box.
        - fontscale (float): The font scale for the text.

        Returns:
        - np.ndarray: The image array with trajectories and bounding boxes of all active tracks.
        """
        #img = np.ascontiguousarray(np.copy(orig_img))
        # if values in dict
        if self.per_class_active_tracks is not None:
            for k in self.per_class_active_tracks.keys():
                active_tracks = self.per_class_active_tracks[k]
                for a in active_tracks:
                    if a.history_observations:
                        if len(a.history_observations) > 2:
                            box = a.history_observations[-1]

                            pitch = 90.0
                            yaw = 90.0
                            roll = 90.0
                            age = -1
                            if face_detect_func is not None:
                                x1, y1 = int(box[0]), int(box[1])
                                x2, y2 = int(box[2]), int(box[3])
                                if (y2 - y1) >= (x2 - x1) * 2:
                                    y2 = y1 + (y2 - y1) / 2
                                person_crop = orig_img[y1:y2, x1:x2]  # 剪切出人物区域
                                pitch, yaw, roll, age = face_detect_func(person_crop)

                            img = self.plot_box_on_img_copy(orig_img, box, a.conf, a.cls, a.id, thickness, fontscale)

                            target_info = next(
                                (target for target in self.target_list if target.get('track_id') == a.id),
                                None)
                            if target_info:
                                if abs(target_info['Pitch'] * 0.1) + abs(target_info['Yaw'] * 0.7) + abs(
                                        target_info['Roll'] * 0.2) > abs(pitch * 0.1) + abs(yaw * 0.7) + abs(
                                        roll * 0.2):
                                    target_info['Pitch'] = pitch
                                    target_info['Yaw'] = yaw
                                    target_info['Roll'] = roll
                                if age > -1:
                                    target_info['Age'] = (age + target_info['Age']) / 2
                                video_writer = target_info['vid_writer']
                                video_writer.write(img)
                                if frames is not None:
                                    frames_seq = target_info['frames_seq']
                                    if frames != target_info['prev'] + 1:
                                        frames_seq.append([target_info['start'], target_info['prev']])
                                        target_info['start'] = frames
                                    target_info['prev'] = frames
                            else:
                                if age < 0:
                                    age = 0
                                new_target = {'Pitch': pitch, 'Yaw': yaw, 'Roll': roll, 'track_id': a.id, 'Age': age}
                                target_video_path = f"{custom_target_path}/target_{a.id}.mp4"
                                new_target['path'] = target_video_path
                                width = orig_img.shape[1]
                                height = orig_img.shape[0]
                                video_writer = cv2.VideoWriter(
                                    target_video_path, cv2.VideoWriter_fourcc(*"mp4v"), 25, (int(width), int(height))
                                )
                                new_target['vid_writer'] = video_writer
                                video_writer.write(img)
                                if frames is not None:
                                    new_target['frames_seq'] = []
                                    new_target['prev'] = frames
                                    new_target['start'] = frames
                                    #new_target['frames_seq'].append([frames, frames])
                                self.target_list.append(new_target)

        else:
            for a in self.active_tracks:
                if a.history_observations:
                    if len(a.history_observations) > 2:
                        box = a.history_observations[-1]
                        pitch = 90.0
                        yaw = 90.0
                        roll = 90.0
                        age = 0
                        if face_detect_func is not None:
                            x1, y1 = int(box[0]), int(box[1])
                            x2, y2 = int(box[2]), int(box[3])
                            if (y2 - y1) >= (x2 - x1) * 2:
                                y2 = int(y1 + (y2 - y1) / 2)
                            person_crop = orig_img[y1:y2, x1:x2]  # 剪切出人物区域
                            pitch, yaw, roll, age = face_detect_func(person_crop)
                        img = self.plot_box_on_img_copy(orig_img, box, a.conf, a.cls, a.id, thickness, fontscale)
                        target_info = next((target for target in self.target_list if target.get('track_id') == a.id),
                                           None)
                        if target_info:
                            if abs(target_info['Pitch'] * 0.1) + abs(target_info['Yaw'] * 0.7) + abs(
                                    target_info['Roll'] * 0.2) > abs(pitch * 0.1) + abs(yaw * 0.7) + abs(roll * 0.2):
                                target_info['Pitch'] = pitch
                                target_info['Yaw'] = yaw
                                target_info['Roll'] = roll
                            if age > -1:
                                target_info['Age'] = (age + target_info['Age']) / 2
                            video_writer = target_info['vid_writer']
                            video_writer.write(img)
                            if frames is not None:
                                frames_seq = target_info['frames_seq']
                                if frames != target_info['prev'] + 1:
                                    frames_seq.append([target_info['start'], target_info['prev']])
                                    target_info['start'] = frames
                                target_info['prev'] = frames
                        else:
                            if age < 0:
                                age = 0
                            new_target = {'Pitch': pitch, 'Yaw': yaw, 'Roll': roll, 'track_id': a.id, 'Age': age}
                            target_video_path = f"{custom_target_path}/target_{a.id}.mp4"
                            new_target['path'] = target_video_path
                            width = orig_img.shape[1]
                            height = orig_img.shape[0]
                            video_writer = cv2.VideoWriter(
                                target_video_path, cv2.VideoWriter_fourcc(*"mp4v"), 25, (int(width), int(height))
                            )
                            new_target['vid_writer'] = video_writer
                            video_writer.write(img)
                            if frames is not None:
                                new_target['frames_seq'] = []
                                new_target['prev'] = frames
                                new_target['start'] = frames
                                #new_target['frames_seq'].append([frames, frames])
                            self.target_list.append(new_target)

    def check_target(self, face_angle_threshold=89, age_threshold=(0, 100)):
        if len(self.target_list) > 0:
            for i, target, in enumerate(self.target_list):
                pitch = abs(target['Pitch'])
                yaw = abs(target['Yaw'])
                roll = abs(target['Roll'])
                age = target['Age']
                print("", target['track_id'], pitch, yaw, roll, age)
                b_age_err = False
                if age > age_threshold[1] or age < age_threshold[0]:
                    b_age_err = True
                if b_age_err or pitch >= face_angle_threshold or yaw >= face_angle_threshold or roll >= face_angle_threshold:
                    video_writer = target['vid_writer']
                    video_writer.release()
                    target_video_path = target['path']
                    if 'frames_seq' in target:
                        del target['frames_seq']
                    try:
                        os.remove(target_video_path)  # 删除指定路径的文件
                        print(f"文件 {target_video_path} 已被删除")
                    except FileNotFoundError:
                        print(f"文件 {target_video_path} 不存在")
                    except PermissionError:
                        print(f"没有权限删除文件 {target_video_path}")
                    except Exception as e:
                        print(f"删除文件时出错: {e}")

    # def get_frame_timecodes(self,video_path) -> list[tuple[int, float]]:
    #     """使用FFprobe获取精确帧时间戳"""
    #     cmd = [
    #         'ffprobe', '-v', 'quiet',
    #         '-print_format', 'json',
    #         '-show_frames',
    #         '-select_streams', 'v:0',
    #         '-show_entries', 'frame=pts_time,pkt_pts_time,coded_picture_number',
    #         video_path
    #     ]
    #     try:
    #         result = subprocess.run(cmd, check=True,
    #                                 capture_output=True, text=True)
    #         frames = json.loads(result.stdout)['frames']
    #         print(frames)
    #         return [
    #             (int(f['coded_picture_number']), float(f['pkt_pts_time']))
    #             for f in frames if f.get('media_type') == 'video'
    #         ]
    #     except Exception as e:
    #         print(f"FFprobe解析失败，使用OpenCV默认帧率: {e}")
    #         return []

    def get_frame_timecodes(self, video_path) -> list[tuple[int, float]]:
        """使用FFprobe获取精确帧时间戳"""
        cmd = [
            'ffprobe', '-v', 'quiet',
            '-print_format', 'json',
            '-show_frames',
            '-select_streams', 'v:0',
            '-show_entries', 'frame=pts_time,pkt_pts_time,coded_picture_number',
            video_path
        ]
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            frames = json.loads(result.stdout)['frames']
            return [
                (int(f['coded_picture_number']), float(f['pkt_pts_time']))
                for f in frames if 'pkt_pts_time' in f and 'coded_picture_number' in f
            ]
        except Exception as e:
            print(f"FFprobe解析失败: {e}")
            return []

    def get_frame_timestamp(self, timecodes,frame_num: int) -> float:
        """获取指定帧的时间戳(秒)"""
        # 可变帧率精确查找
        for num, ts in timecodes:
            if num == frame_num:
                return ts
        return -1.0
    # def out_target_json(self,video_path=None):
    #     data = {
    #         "status": 200,
    #         "message": "succeed",
    #         "data": []
    #     }
    #     frame2timestamp=False
    #     cap = None
    #     if video_path is not None:
    #         cap = cv2.VideoCapture(video_path,cv2.CAP_FFMPEG)
    #     if cap and cap.isOpened():
    #         frame2timestamp=True
    #         fps = cap.get(cv2.CAP_PROP_FPS)
    #         print(fps)
    #         cap.set(cv2.CAP_PROP_FORMAT, -1)  # 原始数据包
    #     if len(self.target_list) > 0:
    #         data = {
    #             "status": 200,
    #             "message": "succeed",
    #             "data": []
    #         }
    #         for i, target, in enumerate(self.target_list):
    #             if 'frames_seq' in target:
    #                 frames_seq = target['frames_seq']
    #                 frames_seq.append([target['start'], target['prev']])
    #                 if frame2timestamp is True:
    #                     for index in range(len(frames_seq)):
    #                         cap.set(cv2.CAP_PROP_POS_FRAMES, frames_seq[index][0])
    #                         # 获取时间戳(毫秒)
    #                         start_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
    #                         cap.set(cv2.CAP_PROP_POS_FRAMES, frames_seq[index][1])
    #                         # 获取时间戳(毫秒)
    #                         prev_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
    #                         frames_seq[index][0] = start_timestamp
    #                         frames_seq[index][1] = prev_timestamp
    #                         #frames_seq.append([start_timestamp, prev_timestamp])
    #                 # else:
    #                 #     frames_seq.append([target['start'], target['prev']])
    #                 sub_target = {
    #                     "label": "target_"+str(i),
    #                     "pos":frames_seq
    #                 }
    #                 data["data"].append(sub_target)
    #     if cap and cap.isOpened():
    #         cap.release()
    #     return data

    def out_target_json(self,video_path=None):
        data = {
            "status": 200,
            "message": "succeed",
            "data": []
        }
        frame2timestamp=False
        timecodes = None
        if video_path is not None:
            timecodes = self.get_frame_timecodes(video_path)
            if timecodes:
                frame2timestamp=True
            # print(timecodes)
        if len(self.target_list) > 0:
            data = {
                "status": 200,
                "message": "succeed",
                "data": []
            }
            for i, target, in enumerate(self.target_list):
                if 'frames_seq' in target:
                    frames_seq = target['frames_seq']
                    frames_seq.append([target['start'], target['prev']])
                    if frame2timestamp is True:
                        for index in range(len(frames_seq)):
                            # 获取时间戳(毫秒)
                            start_timestamp = self.get_frame_timestamp(timecodes,frames_seq[index][0])
                            # 获取时间戳(毫秒)
                            prev_timestamp  = self.get_frame_timestamp(timecodes,frames_seq[index][1])
                            frames_seq[index][0] = start_timestamp
                            frames_seq[index][1] = prev_timestamp
                            #frames_seq.append([start_timestamp, prev_timestamp])
                    # else:
                    #     frames_seq.append([target['start'], target['prev']])
                    sub_target = {
                        "label": "target_"+str(i),
                        "pos":frames_seq
                    }
                    data["data"].append(sub_target)

        return data

