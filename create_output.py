import csv
import pixel_utils
import cv2
import os
from pymediainfo import MediaInfo
def tracks_to_csv(txt_file):
    file1 = open(txt_file, "r+")
    # Read from file log and write to nLog file
    with open(txt_file + ".csv", 'w', newline='') as csv_file:
        for line in file1:
            line = line.replace(' ', ',')
            csv_file.write(line)
            csv_file.flush


def pixelate_tracks_cluster(video_path, det_path, target_ids):
    video = cv2.VideoCapture(video_path)
    frame_rate = video.get(cv2.CAP_PROP_FPS)
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
                    if id in target_ids:
                        x1, y1 = (int(float(line[2+5*i+1])), int(float(line[2+5*i+2])))
                        x2, y2 = (int(float(line[2+5*i+3])), int(float(line[2+5*i+4])))
                        bbox = [x1,y1,x2,y2]
                        frame = pixel_utils.pixelate_face(frame, bbox)
            out.write(frame)
            """cv2.imshow('face Capture', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            """
            frame_counter += 1
    video.release()
    return path_out


def pixelate_tracks_input(video_path, det_path, labels):
    video = cv2.VideoCapture(video_path)
    frame_rate = video.get(cv2.CAP_PROP_FPS)
    success, frame = video.read()
    height, width, layers = frame.shape
    size = (int(width), int(height))
    f_name, f_ext = os.path.splitext(os.path.basename(video_path))
    output_path = os.path.dirname(video_path)
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
                    if labels[id] > -1:
                        x1 = int(float(line[2+5*i+1]))
                        y1 = int(float(line[2+5*i+2]))
                        x2 = x1 + (int(float(line[2+5*i+3])))
                        y2 = y1 + (int(float(line[2+5*i+4])))
                        bbox = [x1,y1,x2,y2]
                        frame = pixel_utils.pixelate_face(frame, bbox)
            out.write(frame)
            frame_counter += 1
    video.release()
    return path_out

#add audio with original videos audio
def add_audio(pix_vid_mute, original_vid, output_dir):
    f_name, f_ext = os.path.splitext(os.path.basename(original_vid))
    os.system("ffmpeg.exe -i " + original_vid + " -vn -c:a libmp3lame -q:a 1 " + original_vid + ".mp3")
    os.system("ffmpeg -i " + pix_vid_mute + " -i " + original_vid + ".mp3 -c copy -map 0:v:0 -map 1:a:0 " + output_dir + "/(pix)" + f_name + ".mp4")
    os.remove(original_vid + ".mp3")
    os.remove(pix_vid_mute)
