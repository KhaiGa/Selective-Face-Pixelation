import csv
import math
import os

import cv2
import numpy as np
from numpy.linalg import norm
import cluster_metrics


def pixelate_face(img, bbox):
    # Get input size
    cropped_img = crop_from_bbox(img, bbox)

    height, width = cropped_img.shape[:2]
    if height < 1 or width < 1:
        return img
    # Desired "pixelated" size
    w, h = (8, 8)

    # Resize input to "pixelated" size
    temp = cv2.resize(cropped_img, (w, h), interpolation=cv2.INTER_LINEAR)

    # Initialize output image
    pixel_img = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)
    y = int(bbox[1])
    h = int(bbox[3] - y)
    x = int(bbox[0])
    w = int(bbox[2] - x)
    img[y:y + h, x:x + w] = pixel_img
    return img


def crop_from_bbox(img, bbox):
    bbox = [0 if i < 0 else i for i in bbox]
    y = int(bbox[1])
    h = int(bbox[3] - y)
    x = int(bbox[0])
    w = int(bbox[2] - x)
    cropped_img = img[y:y + h, x:x + w]
    return cropped_img

#for list of bounding boxes
def scale_bounding_boxes(bounding_box, scale):
    x = bounding_box[0]
    y = bounding_box[1]
    w = bounding_box[2] - x
    h = bounding_box[3] - y
    diff_w = w * scale - w
    diff_h = h * scale - h
    x -= diff_w
    y -= diff_h
    x1 = bounding_box[2] + diff_w
    y1 = bounding_box[3] + diff_h
    rescaled_bbox = np.array([x, y, x1, y1])
    return rescaled_bbox

#for single bounding box
def scale_bounding_box(bounding_box, scale):
    rescaled_bbox = []
    x = bounding_box[0]
    y = bounding_box[1]
    w = bounding_box[2] - x
    h = bounding_box[3] - y
    diff_w = w * scale - w
    diff_h = h * scale - h
    x -= diff_w
    y -= diff_h
    x1 = bounding_box[2] + diff_w
    y1 = bounding_box[3] + diff_h
    rescaled_bbox = np.array([x, y, x1, y1])
    return rescaled_bbox


def compute_sim(emb1, emb2):
    emb1 = emb1.flatten()
    emb2 = emb2.flatten()
    sim = np.dot(emb1, emb2) / (norm(emb1) * norm(emb2))
    return sim

def check_track_continuation(embeddings, track_list, bbox, frame_nr):
    max_sim = 0
    best_match = None
    for track in track_list:
        # check if track end is close to current frame nr
        if abs(frame_nr - track.last_seen_frame) < 10:
            iou = bb_intersection_over_union(bbox, track.last_seen_bbox)
            sim = compute_sim(embeddings, track.avg_emb) + 0.5*iou
            if sim > max_sim:
                max_sim = sim
                best_match = track
    if max_sim > 0.4:
        best_match.last_seen_bbox = bbox
        best_match.last_seen_frame = frame_nr
        return best_match, max_sim
    else:
        return None, None
def check_emb_match(embeddings, person_list):
    max_sim = 0
    best_match = None
    for person in person_list:
        sim = compute_sim(embeddings, person.embeddings)
        if sim > max_sim:
            max_sim = sim
            best_match = person
    if max_sim > 0.6:
        return best_match, max_sim
    else:
        return None, None
def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

