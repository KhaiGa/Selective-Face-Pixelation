import csv
import os
import cv2
import h5py
import numpy as np
import utils
import insightface
import Track

# landmarks have size (5,2)
def get_face_embeddings(img, landmarks, model):
    img = insightface.utils.face_align.norm_crop(img, landmarks)
    embeddings = model.get_embedding(img)
    # print(model.compute_sim(face1, face2))
    return embeddings


def create_tracks_and_emb_file(video_path, meta_path):
    emb_model = insightface.model_zoo.get_model('arcface_r100_v1')
    emb_model.prepare(ctx_id=0)
    print('Starting feature extraction.')
    video = cv2.VideoCapture(video_path)
    faces = []
    current_frame_number = None
    success, frame = video.read()
    f_name, f_ext = os.path.splitext(os.path.basename(video_path))
    os.mkdir(meta_path + "/track_embs")
    os.mkdir(meta_path + "/track_imgs")
    det_path = os.path.join(meta_path, f_name + "(det).csv")
    hdf_path = os.path.join(meta_path, f_name + '.h5')
    tracks_csv = os.path.join(meta_path, f_name + '_tracks.csv')
    if os.path.exists(tracks_csv):
        os.remove(tracks_csv)
    #save embeddings in h5 files
    with h5py.File(hdf_path, 'w') as emb_hdf:
        #save tracks in csv file
        with open(tracks_csv, 'a+') as tracks_file:
            datasets_counter = 0
            # 0-1000
            emb_counter = 0
            emb_list = []
            # list of all individual persons detected
            active_track_list = []
            archived_track_list = []
            id_counter = 0
            with open(det_path) as f:
                csvFile = csv.reader(f)
                for line in csvFile:
                    # line is always 1 ahead of Video
                    next_frame_number, *_ = line
                    next_frame_number = int(next_frame_number)

                    if next_frame_number == current_frame_number:
                        faces.append(line)
                    else:
                        #print(current_frame_number)
                        tracks_line = [current_frame_number, len(faces)]
                        if len(faces) > 0:
                            for face in faces:
                                bbox = [int(float(face[1])), int(float(face[2])), int(float(face[3])), int(float(face[4]))]
                                bbox = [0 if i < 0 else i for i in bbox]
                                landmarks = [[int(float(face[6])), int(float(face[7]))], [int(float(face[8])), int(float(face[9]))],
                                             [int(float(face[10])), int(float(face[11]))], [int(float(face[12])), int(float(face[13]))],
                                             [int(float(face[14])), int(float(face[15]))]]
                                landmarks = np.asarray(landmarks)
                                # get embeddings
                                embs = (get_face_embeddings(frame, landmarks, emb_model))[0]
                                # embs.cpu().detach().numpy()
                                # check for matches
                                id_match, confidence_score = utils.check_track_continuation(embs, active_track_list, bbox,
                                                                                    current_frame_number)
                                #id_match, confidence_score = utils.check_emb_match2(embs, track_list, bbox,
                                #                                                            current_frame_number)
                                #only draw bbox around current face for sample img
                                frame_temp = np.copy(frame)
                                cv2.rectangle(frame_temp, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)

                                #cv2.rectangle(frame_temp, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                                if id_match is not None:
                                    id_match.update_embeddings_avg(embs)
                                    id_match.update_embeddings_list(embs)
                                    #new_sim = utils.compute_sim(embs, id_match.avg_emb)
                                    bbox_large = utils.scale_bounding_boxes(bbox, 2)
                                    sample_img = utils.crop_from_bbox(frame_temp, bbox_large)
                                    id_match.update_sample_img(sample_img, embs, landmarks)
                                    if(id_match.sample_img is None):
                                        print("error: no image")
                                else:
                                    # create new track
                                    bbox_large = utils.scale_bounding_boxes(bbox, 2)
                                    sample_img = utils.crop_from_bbox(frame_temp, bbox_large)
                                    new_track = Track.Track(id_counter, embs, sample_img, bbox, current_frame_number)
                                    new_track.update_sample_img(sample_img, embs, landmarks)
                                    #put into init
                                    new_track.avg_emb = embs
                                    id_counter += 1
                                    active_track_list.append(new_track)
                                    #remove?
                                    id_match = new_track
                                # draw bbox with color
                                """cv2.putText(frame, 'id:' + str(id_match.id),
                                            (bbox[0], bbox[1]),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            1,
                                            (255, 255, 255),
                                            2)"""
                                #add track information to csv line in order: frame, # of tracks, track id, x, y, w, h, track id_2, x_2, y_2 ,...
                                width = bbox[2] - bbox[0]
                                height = bbox[3] - bbox[1]
                                values = [str(id_match.id), bbox[0], bbox[1], width, height]
                                tracks_line += values
                                if emb_counter == 1000:
                                    emb_hdf.create_dataset('dataset_' + str(datasets_counter), data=emb_list)
                                    emb_list = []
                                    emb_counter = 0
                                    datasets_counter += 1
                                else:
                                    embeddings = np.insert(embs, 0, current_frame_number)
                                    emb_list.append(embeddings)
                                    emb_counter += 1
                        for track in active_track_list:
                            if track.last_seen_frame < current_frame_number - 10:
                                archived_track_list.append(track)
                                active_track_list.remove(track)
                        print('archived len : ', len(archived_track_list))
                        print('active len : ', len(active_track_list))
                        tracks_line = ','.join(map(str, tracks_line))
                        if current_frame_number is not None:
                            print('tracksLine', tracks_line)
                            tracks_file.write(tracks_line)
                            tracks_file.write('\n')
                            tracks_file.flush
                        #print(len(track_list))
                        if len(line) > 1:
                            faces = [line]
                        success, frame = video.read()

                    current_frame_number = next_frame_number
                # end of csv file
                #track_embs_path = output_dir + "/track_embs"
                #save
                archived_track_list += active_track_list
                sample_img_features = []
                for track in archived_track_list:
                    #only take into account tracks with more than 5 entries for testing to reduce false tracks
                    if len(track.emb_list) >= 5:
                        # add 0 in front of track id, max 99999
                        id_name = "0" * (6 - len(str(track.id))) + str(track.id)
                        #add sample img embedding to list
                        sample_img_features.append(track.sample_img_emb)
                        # save embeddings
                        np.save(os.path.join(meta_path, "track_embs", id_name), track.emb_list)
                        # save sample_img
                        print(track.sample_img.shape)
                        cv2.imwrite(os.path.join(meta_path, "track_imgs", id_name + '.jpg'), track.sample_img)
                        #HAC clustering on tracks
                np.save(os.path.join(meta_path, "img_features"), sample_img_features)
                emb_hdf.create_dataset('dataset_' + str(datasets_counter), data=emb_list)
        video.release()
        return tracks_csv

#create_tracks_and_emb_file('Videos/5102_KK_20200219_12_Busy.mp4', 'Videos/5102_KK_20200219_12_Busy/')