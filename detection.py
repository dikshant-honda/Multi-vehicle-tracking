#! usr/bin/env python3

import sys
import signal
import argparse
import time
from pathlib import Path

import numpy as np
import math
from collections import deque

import rospy
from multi_vehicle_tracking.msg import pos_and_vel, queue
from visualization_msgs.msg import Marker

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
import matplotlib.pyplot as plt
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from lane import lanes


def estimateSpeed(location1, location2):
    d_pixels = math.sqrt(math.pow(
        location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
    ppm = 15  # Pixels per Meter
    d_meters = d_pixels / ppm
    time_constant = 30 * 3.6
    speed = d_meters * time_constant
    return int(speed)


def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))


def detect(save_img=True):
    names, source, weights, view_img, save_txt, imgsz, trace = opt.names, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith(
        '.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name,
                    exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True,
                                                          exist_ok=True)  # make dir
    
    # lane information
    mid_lanes = lanes()

    # initialize deepsort
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
    deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                        max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT, nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load(
            'weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = load_classes(names)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
            next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(
                ), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + \
                ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            # normalization gain whwh
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string
                xywh_bboxs = []
                confs = []
                oids = []

                for *xyxy, conf, cls in reversed(det):
                    x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
                    xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                    xywh_bboxs.append(xywh_obj)
                    confs.append([conf.item()])
                    oids.append(int(cls))

                xywhs = torch.Tensor(xywh_bboxs)
                confss = torch.Tensor(confs)
                outputs = deepsort.update(xywhs, confss, oids, im0)
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -2]
                    object_id = outputs[:, -1]
                    # DrawBoxes function to draw the bounding boxes, label them and show the ID of the tracker
                    draw_boxes(im0, mid_lanes, bbox_xyxy, names, object_id,
                               save_txt, txt_path, identities)

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(
                        f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")


def mark(marker, id, type, x, y, z, w, color):
    marker.header.frame_id = "map"
    marker.header.stamp = rospy.get_rostime()
    marker.ns = "vehicle"
    marker.id = id
    marker.type = type
    marker.action = 0
    marker.pose.position.x = x
    marker.pose.position.y = y
    marker.pose.position.z = z
    marker.pose.orientation.x = 0
    marker.pose.orientation.y = 0
    marker.pose.orientation.z = 0
    marker.pose.orientation.w = w
    marker.scale.x = 50.0
    marker.scale.y = 50.0
    marker.scale.z = 50.0

    if color == "r":
        marker.color.r = 1.0
        marker.color.g = 0.0
    else:
        marker.color.r = 0.0
        marker.color.g = 1.0
    marker.color.b = 0.0
    marker.color.a = 1.0
    marker.lifetime.nsecs = 75000000
    markerPub.publish(marker)


def ros_updates(env_info, id, transformed_center, transformed_center_deque, mid_lanes):
    pos_and_vel_data = pos_and_vel()
    robotMarker = Marker()

    if transformed_center_deque[id][-1][1]-transformed_center_deque[id][0][1] >= 0:
        direction = 1
        mark(robotMarker, id, 1,
             transformed_center[0][0][0], transformed_center[0][0][1], 0, 1, "r")

    else:
        direction = -1
        mark(robotMarker, id, 2,
             transformed_center[0][0][0], transformed_center[0][0][1], 0, -1, "g")

    pos_and_vel_data.id.data = id
    pos_and_vel_data.x_position.data = transformed_center[0][0][0]
    pos_and_vel_data.y_position.data = transformed_center[0][0][1]
    pos_and_vel_data.speed.data = sum(
        speed_line_queue[id][-5:]) // len(speed_line_queue[id][-5:])
    pos_and_vel_data.direction.data = direction
    env_info.info.append(pos_and_vel_data)
    d = math.inf

    for line in mid_lanes:
        p1 = np.array([line[0], line[1]])
        p2 = np.array([line[2], line[3]])
        p3 = np.array([transformed_center[0][0][0],
                      transformed_center[0][0][1]])
        dist = abs(np.cross(p2-p1, p3-p1)/np.linalg.norm(p2-p1))
        if dist < d:
            pos_and_vel_data.start_x.data = line[0]
            pos_and_vel_data.start_y.data = line[1]
            pos_and_vel_data.end_x.data = line[2]
            pos_and_vel_data.end_y.data = line[3]
            d = dist

    # publishing position and velocity data of every vehicle in the frame
    if len(env_info.info) == len(data_deque):
        env_info_pub.publish(env_info)

    rate.sleep()


def draw_boxes(img, mid_lanes, bbox, names, object_id, save_txt, txt_path, identities=None, offset=(0, 0)):
    env_info = queue()
    for key in list(data_deque):
        if key not in identities:
            data_deque.pop(key)

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        # find center of bottom edge of the bounding box
        center = (int((x2+x1) / 2), int((y2+y2)/2))
        np_center = np.array(center, dtype=np.float32).reshape(1, -1, 2)
        # transformed_center = cv2.perspectiveTransform(np_center, M)
        transformed_center = np.flip(cv2.perspectiveTransform(np_center, M), 2)

        # get a unique ID of each object
        id = int(identities[i]) if identities is not None else 0

        if id not in data_deque:
            data_deque[id] = deque(maxlen=opt.trailslen)
            transformed_center_deque[id] = deque(maxlen=64)
            speed_line_queue[id] = []
            heading_deque[id] = deque(maxlen=64)
            beta[id] = []

        # Setting a unique color for each object bounding box and rounded rectangle which contains the label
        color = compute_color_for_labels(object_id[i])
        obj_name = names[object_id[i]]
        # Setting the Label in the Required Format
        label = '{}{:d}'.format("", id) + ":" + '%s' % (obj_name)

        data_deque[id].appendleft(center)
        transformed_center_deque[id].appendleft(transformed_center[0][0])

        if len(data_deque[id]) >= 2:

            if len(speed_line_queue[id]) >= 1:
                A = 0.05 * \
                    estimateSpeed(
                        transformed_center_deque[id][1], transformed_center_deque[id][0]) + 0.95 * speed_line_queue[id][-1]
                B = 0.05 * (A - speed_line_queue[id][-1]) + 0.95 * beta[id][-1]
                beta[id].append(B)
                obj_speed = A+B
            else:
                obj_speed = estimateSpeed(
                    transformed_center_deque[id][1], transformed_center_deque[id][0])
                beta[id].append(1)
            speed_line_queue[id].append(obj_speed)

            # publish the data
            ros_updates(env_info, id, transformed_center,
                        transformed_center_deque, mid_lanes)

        try:
            # label = label + " " + str(speed_line_queue[id][-1]) + "km/h"  ##
            label = label + " " + \
                str(sum(speed_line_queue[id][-5:]) //
                    len(speed_line_queue[id][-5:])) + "km/h"
        except:
            pass

        UI_box(box, img, label=label, color=color, line_thickness=2)

        # draw trail
        for i in range(1, len(data_deque[id])):
            # check if on buffer value is none
            if data_deque[id][i - 1] is None or data_deque[id][i] is None:
                continue
            # generate dynamic thickness of trails
            thickness = int(np.sqrt(opt.trailslen / float(i + i)) * 1.5)
            # draw trails
            cv2.line(img, data_deque[id][i - 1],
                     data_deque[id][i], color, thickness)

        if save_txt and len(data_deque[id]) >= 2:
            line = (id, x1, y1, sum(speed_line_queue[id][-5:]) //
                    len(speed_line_queue[id][-5:]))
            with open(txt_path + '.txt', 'a') as f:
                f.write(('%g ' * len(line)).rstrip() % line + '\n')

    return img


def xyxy_to_xywh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def compute_color_for_labels(label):
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    """
    Simple function that adds fixed color depending on the class
    """
    if label == 0:  # person
        color = (85, 45, 255)
    elif label == 2:  # Car
        color = (222, 82, 175)
    elif label == 3:  # Motorbike
        color = (0, 204, 255)
    elif label == 5:  # Bus
        color = (0, 149, 255)
    else:
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1, y1 = pt1
    x2, y2 = pt2
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)

    cv2.circle(img, (x1 + r, y1+r), 2, color, 12)
    cv2.circle(img, (x2 - r, y1+r), 2, color, 12)
    cv2.circle(img, (x1 + r, y2-r), 2, color, 12)
    cv2.circle(img, (x2 - r, y2-r), 2, color, 12)

    return img


def UI_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(
        0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 5, thickness=tf)[0]

        img = draw_border(img, (c1[0], c1[1] - t_size[1] - 3),
                          (c1[0] + t_size[0], c1[1]+3), color, 1, 8, 2)

        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 5,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def signal_handler(signal, frame):
    print("\nCtrl+C detected. Exiting!")
    sys.exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='yolov7e6e.pt', help='model.pt path(s)')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str,
                        default='test1.mp4', help='source')
    parser.add_argument('--img-size', type=int, default=1280,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.6, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.4, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',
                        default=True, help='display results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true',
                        help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true',
                        help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument('--update', action='store_true',
                        help='update all models')
    parser.add_argument('--project', default='runs/detect',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--names', type=str,
                        default='data/coco.names', help='*.cfg path')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true',
                        help='don`t trace model')
    parser.add_argument('--trailslen', type=int, default=64,
                        help='trails size (new parameter)')
    opt = parser.parse_args()

    data_deque = {}
    transformed_center_deque = {}
    heading_deque = {}
    beta = {}
    speed_line_queue = {}

    transform_matrix_path = "./birdeyes/transform_matrix.npy"
    M = np.load(transform_matrix_path)
    M = np.array(M, np.float32)

    # publishing the data
    env_info_pub = rospy.Publisher('env_info', queue, queue_size=10)
    markerPub = rospy.Publisher('robotMarker', Marker, queue_size=10)
    rospy.init_node('data')
    rate = rospy.Rate(10000)

    signal.signal(signal.SIGINT, signal_handler)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
