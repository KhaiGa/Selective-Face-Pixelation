from __future__ import print_function
import os
import argparse
import insightface
import cv2
import sys


def create_det_file(video_path, meta_path):
    model = insightface.model_zoo.get_model('retinaface_r50_v1')
    model.prepare(ctx_id=0, nms=0.4)
    cap = cv2.VideoCapture(video_path)
    total_frame_nr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_nr = 0
    f_name, f_ext = os.path.splitext(os.path.basename(video_path))
    csv_f_name = f_name + '(det).csv'
    csv_path = os.path.join(meta_path, csv_f_name)

    # if csv file already exists, delete it
    if os.path.exists(csv_path):
        os.remove(csv_path)

    success, frame = cap.read()
    with open(csv_path, 'a+') as faces_file:
        while success:
            success, frame = cap.read()
            #print progress
            print(frame_nr, "/", total_frame_nr)
            #sys.stdout.write("\r%d%%" % float(frame_nr/total_frame_nr))
            #sys.stdout.flush()
            if not success:
                break
            all_bboxes, all_landmarks = model.detect(frame, threshold=0.75, scale=1.0)
            if len(all_bboxes) == 0:
                faces_file.write(str(frame_nr))
                faces_file.write('\n')
                faces_file.flush()
            else:
                for bbox, landmarks in zip(all_bboxes, all_landmarks):
                    cv2.rectangle(frame, (int(float(bbox[0])), int(float(bbox[1]))), (int(float(bbox[2])), int(float(bbox[3]))), (0, 0, 255), 2)
                    # save frame, bbox, landmarks to face_detection.csv
                    line = []
                    #x1,y1,x2,y2, det_score
                    for value in bbox:
                        line.append(value)
                    for (x,y) in landmarks:
                        line.append(x)
                        line.append(y)
                    line = ','.join(map(str, line))
                    faces_file.write(f'{frame_nr},{line}\n')
                    faces_file.flush()
            """cv2.imshow('frame',frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break"""
            frame_nr += 1
    cap.release()
    return csv_path
