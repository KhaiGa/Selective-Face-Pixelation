import math
import os
import cv2
import numpy as np
from numpy.linalg import norm


def pixelate_face(img, bbox):
    # Get input size
    cropped_img = crop_from_bbox(img, bbox)

    height, width = cropped_img.shape[:2]
    if height < 1 or width < 1:
        return img
    # Desired "pixelated" size, smaller -> stronger pixelation effect
    w, h = (10, 10)

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


def pixelate_with_tracks(video_path, det_path, target_ids):
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = video.get(cv2.CAP_PROP_FPS)
    print('frame_rate', frame_rate)
    faces = []
    current_frame_number = 0
    success, frame = video.read()
    height, width, layers = frame.shape
    size = (int(width), int(height))
    f_name, f_ext = os.path.splitext(os.path.basename(video_path))
    output_path = os.path.dirname(det_path)
    path_out = output_path + '/(no_audio)' + f_name + '.mp4'
    out = cv2.VideoWriter(path_out, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, size)
    frame_counter = 0
    with open(det_path) as f:
        csvFile = csv.reader(f)
        for line in csvFile:
            success, frame = video.read()
            nr_dets = int(float(line[1]))
            if nr_dets != 0:
                for i in range(nr_dets):
                    id = int(line[2+5*i])
                    print("id", id)
                    if id in target_ids:
                        x1, y1 = (int(float(line[2+5*i+1])), int(float(line[2+5*i+2])))
                        x2, y2 = (int(float(line[2+5*i+3])), int(float(line[2+5*i+4])))
                        bbox = [x1,y1,x2,y2]
                        frame = pixelate_face(frame, bbox)
            out.write(frame)
            cv2.imshow('face Capture', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            frame_counter += 1
    video.release()

def crop_from_bbox(img, bbox):
    y = int(bbox[1])
    h = int(bbox[3] - y)
    x = int(bbox[0])
    w = int(bbox[2] - x)
    cropped_img = img[y:y + h, x:x + w]
    return cropped_img

#for list of bounding boxes
def scale_bounding_boxes(bounding_boxes, scale):
    rescaled_bboxes = []
    for index, bounding_box in enumerate(bounding_boxes):
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
        rescaled_bboxes[index] = np.array([x, y, x1, y1])
    return rescaled_bboxes

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
def check_emb_match2(embeddings, person_list, bbox, frame_nr):
    max_sim = 0
    best_match = None
    for person in person_list:
        sim = compute_sim(embeddings, person.embeddings)
        dist = 10000
        # check if person was recently seen (in previous frames) and how close new & old bboxs are
        if abs(frame_nr - person.last_seen_frame) < 5:
            prev_bbox = person.last_seen_bbox
            bbox_width = bbox[2] - bbox[0]
            bbox_height = bbox[3] - bbox[1]
            prev_bbox_width = prev_bbox[2] - prev_bbox[0]
            prev_bbox_height = prev_bbox[3] - prev_bbox[1]
            bbox_center = ((bbox[0] + bbox_width / 2), (bbox[1] + bbox_height / 2))
            prev_bbox_center = ((prev_bbox[0] + prev_bbox_width / 2), (prev_bbox[1] + prev_bbox_height / 2))
            # distance between centers
            dist = math.sqrt((bbox_center[0] - prev_bbox_center[0]) ** 2 + (bbox_center[1] - prev_bbox_center[1]) ** 2)
            sim += 0.2
        if sim > max_sim:
            max_sim = sim
            best_match = person
    if max_sim > 0.4:
        return best_match, max_sim
    else:
        return None, None
def check_emb_match_loc(embeddings, person_list, bbox, frame_nr):
    max_sim = 0
    best_match = None
    for person in person_list:
        dist = 10000
        #check if person was recently seen (in previous frames) and how close new & old bboxs are
        if abs(frame_nr - person.last_seen_frame) < 20:
            prev_bbox = person.last_seen_bbox
            bbox_width = bbox[2] - bbox[0]
            bbox_height = bbox[3] - bbox[1]
            prev_bbox_width = prev_bbox[2] - prev_bbox[0]
            prev_bbox_height = prev_bbox[3] - prev_bbox[1]
            bbox_center = ((bbox[0] + bbox_width/2), (bbox[1] + bbox_height/2))
            prev_bbox_center = ((prev_bbox[0] + prev_bbox_width/2), (prev_bbox[1] + prev_bbox_height/2))
            #distance between centers
            dist = math.sqrt((bbox_center[0] - prev_bbox_center[0]) ** 2 + (bbox_center[1] - prev_bbox_center[1]) ** 2)
            #print(dist)
        sim_sum = 0
        for prev_embeddings in person.embeddings_list:
            sim = compute_sim(embeddings, prev_embeddings)
            sim_sum += sim
        avg_sim = sim_sum / len(person.embeddings_list)
        if dist < 50:
            avg_sim += 0.3
        #sim_diff = abs(avg_sim - max_sim)
        #print(avg_sim)
        #if either new_sim is higher or! new_sim is also close spatially and temporally
        if avg_sim > max_sim:
            max_sim = avg_sim
            best_match = person
    if max_sim > 0.4:
        #update last_seen_bbox and last_seen_frame
        best_match.last_seen_bbox = bbox
        best_match.last_seen_frame = frame_nr
        return best_match, max_sim

    else:
        return None
"""def check_id(source_embs, target_embs):
    #source_embs = torch.cat(embs)
    diff = source_embs.unsqueeze(-1) - target_embs.transpose(1, 0).unsqueeze(0)
    dist = torch.sum(torch.pow(diff, 2), dim=1)
    minimum, min_idx = torch.min(dist, dim=1)
    min_idx[minimum > args.threshold] = -1  # if no match, set idx to -1
    return minimum, min_idx"""


