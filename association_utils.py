import csv
import os
import glob
import cv2
import insightface
import numpy as np
import cluster_metrics
from collections import Counter

from recognition import get_face_embeddings


def simple_read_dataset(video):
    """Simple dataset reading function for purpose of checking evaluation code
    """
    print('Loading dataset:', video)
    label_fname = 'testing/data/ids/' + video + '.ids'
    with open(label_fname, 'r') as fid:
        fid.readline()  # ignore header
        data = fid.readlines()  # track to name
        data = [line.strip().split() for line in data if line.strip()]
        # trackid --> name mapping
        ids = {int(line[0]): line[1] for line in data}
    # get unique names and assign numbers
    uniq_names = list(set(ids.values()))
    uniq_names.sort()
    print('uniq_names: ', uniq_names)
    # Read feature files
    X, y = [], []
    all_feature_fname = glob.glob('testing/data/gt_features/' + video + '/*.npy')
    for fname in all_feature_fname:
        # load and append feature
        feat = np.load(fname)
        X.append(feat.mean(0))
        # append label
        tid = int(os.path.splitext(os.path.basename(fname))[0])
        y.append(uniq_names.index(ids[tid]))

    X = np.array(X)
    y = np.array(y)
    return [X, y, uniq_names]
def get_gt_cluster_index_buffy(uniq_names):
    #if name is not in uniq_names, set gt_cluster_index to -1, else alphabetically
    gt_cluster_index = [-1] * 8
    if 'anya' in uniq_names:
        gt_cluster_index[0] = uniq_names.index('anya')
    if 'buffy' in uniq_names:
        gt_cluster_index[1] = uniq_names.index('buffy')
    if 'dawn' in uniq_names:
        gt_cluster_index[2] = uniq_names.index('dawn')
    if 'riley' in uniq_names:
        gt_cluster_index[3] = uniq_names.index('riley')
    if 'rupert' in uniq_names:
        gt_cluster_index[4] = uniq_names.index('rupert')
    if 'spike' in uniq_names:
        gt_cluster_index[5] = uniq_names.index('spike')
    if 'willow' in uniq_names:
        gt_cluster_index[6] = uniq_names.index('willow')
    if 'xander' in uniq_names:
        gt_cluster_index[7] = uniq_names.index('xander')
    """gt_cluster_index = []
    if 'anya' in uniq_names:
        gt_cluster_index.append(uniq_names.index('anya'))
    if 'buffy' in uniq_names:
        gt_cluster_index.append(uniq_names.index('buffy'))
    if 'dawn' in uniq_names:
        gt_cluster_index.append(uniq_names.index('dawn'))
    if 'marc' in uniq_names:
        gt_cluster_index.append(uniq_names.index('marc'))
    if 'rupert' in uniq_names:
        gt_cluster_index.append(uniq_names.index('rupert'))
    if 'spike' in uniq_names:
        gt_cluster_index.append(uniq_names.index('spike'))
    if 'willow' in uniq_names:
        gt_cluster_index.append(uniq_names.index('willow'))
    if 'xander' in uniq_names:
        gt_cluster_index.append(uniq_names.index('xander'))"""
    return gt_cluster_index
def get_gt_cluster_index_bbt(uniq_names):
    #if name is not in uniq_names, set gt_cluster_index to -1, else alphabetically
    gt_cluster_index = [-1] * 5
    if 'howard' in uniq_names:
        gt_cluster_index[0] = uniq_names.index('howard')
    if 'leonard' in uniq_names:
        gt_cluster_index[1] = uniq_names.index('leonard')
    if 'penny' in uniq_names:
        gt_cluster_index[2] = uniq_names.index('penny')
    if 'raj' in uniq_names:
        gt_cluster_index[3] = uniq_names.index('raj')
    if 'sheldon' in uniq_names:
        gt_cluster_index[4] = uniq_names.index('sheldon')
    return gt_cluster_index
# create new csv tracks file with clustered ids
def create_det_with_ids(track_labels, csv_path):
    f_name, f_ext = os.path.splitext(os.path.basename(csv_path))
    new_csv_name = f_name + "_id.csv"
    results_path = os.path.dirname(csv_path)
    new_csv_path = os.path.join(results_path, new_csv_name)
    if os.path.exists(new_csv_path):
        os.remove(new_csv_path)
    with open(new_csv_path, 'a+') as new_csv:
        with open(csv_path) as f:
            prev_csv = csv.reader(f)
            for line in prev_csv:
                nr_dets = int(float(line[1]))
                if nr_dets != 0:
                    new_line = [line[0], line[1]]
                    for i in range(nr_dets):
                        id = line[2 + 5 * i]
                        # look for assigned cluster ID in track_labels
                        if id in track_labels:
                            new_id = track_labels[id]
                        else:
                            new_id = -1
                        x1, y1 = (int(float(line[2 + 5 * i + 1])), int(float(line[2 + 5 * i + 2])))
                        x2, y2 = (x1 + (int(float(line[2 + 5 * i + 3]))), y1 + (int(float(line[2 + 5 * i + 4]))))
                        new_line += (new_id, x1, y1, x2, y2)
                    new_line = ','.join(map(str, new_line))
                    new_csv.write(new_line)
                    new_csv.write('\n')
                    new_csv.flush()
                else:
                    line = ','.join(map(str, line))
                    new_csv.write(line)
                    new_csv.write('\n')
                    new_csv.flush()


# create new csv tracks file with clustered ids
def create_det_with_ids_original_tracks(track_labels, csv_path):
    track_list = []
    embs_path = os.path.join(os.path.dirname(csv_path), 'track_embs')
    for file in os.listdir(embs_path):
        if '.npy' in file:
            f_name, f_ext = os.path.splitext(os.path.basename(file))
            track_list.append(f_name)
    f_name, f_ext = os.path.splitext(os.path.basename(csv_path))
    new_csv_path = os.path.dirname(csv_path) + "/" + f_name + "_id.csv"
    if os.path.exists(new_csv_path):
        os.remove(new_csv_path)
    with open(new_csv_path, 'a+') as new_csv:
        with open(csv_path) as f:
            prev_csv = csv.reader(f)
            next(prev_csv)
            next(prev_csv)
            for line in prev_csv:
                nr_dets = int(float(line[2]))
                if nr_dets != 0:
                    new_line = [line[0], line[1], line[2]]
                    for i in range(nr_dets):
                        id = int(float(line[3 + 5 * i]))
                        # look for assigned cluster ID in track_labels
                        id_with_0s = "0" * (6 - len(str(id))) + str(id)
                        index = track_list.index(id_with_0s)
                        new_id = track_labels[index]
                        x1, y1 = (int(float(line[3 + 5 * i + 1])), int(float(line[3 + 5 * i + 2])))
                        x2, y2 = (int(float(line[3 + 5 * i + 3])), int(float(line[3 + 5 * i + 4])))
                        new_line += (new_id, x1, y1, x2, y2)
                    new_line = ','.join(map(str, new_line))
                    new_csv.write(new_line)
                    new_csv.write('\n')
                    new_csv.flush()
                else:
                    line = ','.join(map(str, line))
                    new_csv.write(line)
                    new_csv.write('\n')
                    new_csv.flush()


def create_dict_for_labels(track_labels, results_path):
    # create dictionary mapping for track id name -> label
    embs_path = os.path.join(results_path, "track_embs/")
    track_id_list = []
    for file in os.listdir(embs_path):
        if '.npy' in file:
            f_name, f_ext = os.path.splitext(os.path.basename(file))
            if f_name != '000000':
                f_name = f_name.lstrip('0')
            else:
                f_name = '0'
            track_id_list.append(f_name)
    track_dict = {}

    for index, f_name in enumerate(track_id_list):
        track_dict[f_name] = track_labels[index]
    return track_dict


def evaluate_input_labels(y_gt, track_labels, own_cluster_nr):
    identified_ids = []
    # create list of all gt_ids that were identified via images
    for i in range(own_cluster_nr):
        # get list of indices for cluster i
        index_list = [index for index, x in enumerate(track_labels) if x == i]
        gt_label_list = []
        for index in index_list:
            gt_label_list.append(y_gt[index])
        #max_occ_id = max(gt_label_list, key=gt_label_list.count)
        occurence_count = Counter(gt_label_list)
        max_occ_id = occurence_count.most_common(1)[0]
        #get nr of all ground truth tracks that match highest ID
        gt_cluster_list = [index for index, id in enumerate(y_gt) if id == max_occ_id[0]]
        accuracy = max_occ_id[1]/len(gt_cluster_list)
        identified_ids.append(max_occ_id)
    #print(identified_ids)

def evaluate_input_PR(y_gt, gt_cluster_index, track_labels, cluster_nr):
    sum_precision = 0
    sum_recall = 0
    real_cluster_nr = len([i for i in gt_cluster_index if i >= 0])
    print('real_cluster_nr', real_cluster_nr)
    # create list of all gt_ids that were identified via images
    for i in range(cluster_nr):
        if gt_cluster_index[i] == -1:
            print(i, ' not in episode')
            continue
        gt_cluster_id = gt_cluster_index[i]
        # get list of indices for cluster i
        selected_index_list = [index for index, x in enumerate(track_labels) if x == i]
        if len(selected_index_list) == 0:
            print("skipping cluster ", i)
            continue
        """selected_label_list = []
        for index in selected_index_list:
            selected_label_list.append(y_gt[index])"""
        gt_positives = [index for index, id in enumerate(y_gt) if id == gt_cluster_id]
        true_positives = len(list(set(selected_index_list) & set(gt_positives)))
        false_negatives = len(gt_positives) - true_positives
        false_positives = len(selected_index_list) - true_positives
        print('cluster ', i, ', gt_cluster_id = ', gt_cluster_id, 'false negatives: ', false_negatives, 'false positives: ', false_positives)
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        sum_precision += precision
        sum_recall += recall
        print(precision, recall)
    avg_precision = sum_precision / real_cluster_nr
    avg_recall = sum_recall / real_cluster_nr
    return avg_precision, avg_recall

def evaluate_cluster_PR(y_gt, gt_cluster_index, track_labels, selected_clusters):
    sum_precision = 0
    sum_recall = 0
    nr_of_occuring_chars = len([i for i in gt_cluster_index if i >= 0])
    # create list of all gt_ids that were identified via images
    clusters_not_found = []
    for i, cluster in enumerate(selected_clusters):
        if len(cluster) < 1:
            print(i, ' no cluster found')
            clusters_not_found.append(i)
            continue
        #skip if either the person doesn't appear in episode or not in selected clusters
        if gt_cluster_index[i] == -1 or cluster[0] == -1:
            print(i, ' not in Episode or in cluster')
            continue
        gt_cluster_id = gt_cluster_index[i]
        # get list of indices for cluster i
        selected_index_list = [index for index, x in enumerate(track_labels) if x in cluster]
        if len(selected_index_list) == 0:
            print("No tracks with cluster ", i)
            continue
        gt_positives = [index for index, id in enumerate(y_gt) if id == gt_cluster_id]
        true_positives = len(list(set(selected_index_list) & set(gt_positives)))
        false_negatives = len(gt_positives) - true_positives
        false_positives = len(selected_index_list) - true_positives
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        print('cluster ', cluster, ', gt_cluster_id = ', gt_cluster_id, 'precision', precision, 'recall ', recall)
        sum_precision += precision
        sum_recall += recall
    avg_precision = sum_precision / nr_of_occuring_chars
    avg_recall = sum_recall / nr_of_occuring_chars
    return avg_precision, avg_recall, clusters_not_found, nr_of_occuring_chars

def evaluate_input_PR_with_voting(y_gt, track_labels, own_cluster_nr):
    sum_precision = 0
    sum_recall = 0
    # create list of all gt_ids that were identified via images
    for i in range(own_cluster_nr):
        # get list of indices for cluster i
        selected_index_list = [index for index, x in enumerate(track_labels) if x == i]
        if len(selected_index_list) == 0:
            print("skipping cluster ", i)
            continue
        selected_label_list = []
        for index in selected_index_list:
            selected_label_list.append(y_gt[index])
        #max_occ_id = max(gt_label_list, key=gt_label_list.count)
        occurence_count = Counter(selected_label_list)
        gt_id = occurence_count.most_common(1)[0][0]
        gt_positives = [index for index, id in enumerate(y_gt) if id == gt_id]
        true_positives = occurence_count.most_common(1)[0][1]
        false_negatives = len(gt_positives) - true_positives
        false_positives = len(selected_label_list) - true_positives
        print(occurence_count.most_common(2))
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        sum_precision += precision
        sum_recall += recall
    avg_precision = sum_precision / own_cluster_nr
    avg_recall = sum_recall / own_cluster_nr
    print(avg_precision, avg_recall)
    return avg_precision, avg_recall



def check_missing_clusters(y_gt, track_labels, video_name, gt_cluster_nr, own_cluster_nr):
    missing_clusters = [i for i in range(gt_cluster_nr)]
    # remove cluster id with highest count for every detected cluster from missing_clusters
    for i in range(own_cluster_nr):
        # get list of indices for cluster i
        index_list = [index for index, x in enumerate(track_labels) if x == i]
        gt_label_list = []
        for index in index_list:
            gt_label_list.append(y_gt[index])
        max_occ_id = max(gt_label_list, key=gt_label_list.count)
        if max_occ_id in missing_clusters:
            missing_clusters.remove(max_occ_id)
    print("missing: ", missing_clusters)


def evaluate_clusters(y_gt, labels):
    wcp = cluster_metrics.weighted_purity(y_gt, labels)
    nmi = cluster_metrics.NMI(y_gt, labels)
    return wcp, nmi


def get_track_list(embs_path):
    track_list = []
    for file in os.listdir(embs_path):
        if '.npy' in file:
            f_name, f_ext = os.path.splitext(os.path.basename(file))
            track_list.append(f_name)
    return track_list


# gets ground truth tracks for Video, (removes not detected tracks)
def get_gt_tracks(output_dir, video):
    track_list = get_track_list(os.path.join(output_dir, 'track_embs'))
    X, y_gt, uniq_names = simple_read_dataset(video)
    all_feature_fname = glob.glob('testing/data/gt_features/' + video + '/*.npy')
    all_feature_fname = [os.path.basename(file) for file in all_feature_fname]
    y_gt_new = []

    missing_tracks = []
    for index, f_name in enumerate(all_feature_fname):
        if f_name[:-4] not in track_list:
            print("removing", f_name[:-4])
            missing_tracks.append(f_name[:-4])
        else:
            y_gt_new.append(y_gt[index])
    y_gt_new = np.array(y_gt_new)
    return y_gt_new, missing_tracks, uniq_names
def get_gt_tracks_own(output_dir):
    pass

def create_npy_from_imgs(img_path, output_path):
    dir = os.path.dirname(img_path)
    target_emb_list = []
    img_features_path = os.path.join(output_path, 'img_features')
    print('Extracting features from input images')
    det_model = insightface.model_zoo.get_model('retinaface_r50_v1')
    det_model.prepare(ctx_id=0, nms=0.4)
    emb_model = insightface.model_zoo.get_model('arcface_r100_v1')
    emb_model.prepare(ctx_id=0)
    if not os.path.exists(img_features_path):
        os.mkdir(img_features_path)
    for file in os.listdir(img_path):
        print(file)
        if not os.path.exists(os.path.join(img_features_path, file + '.npy')):
            try:
                img = cv2.imread(os.path.join(img_path, file))
            except IOError:
                print("Image folder must contain images only")
            bbox, landmark = det_model.detect(img, threshold=0.5, scale=1.0)
            if len(bbox) == 0:
                print("found no faces on ", file)
                bbox, landmark = det_model.detect(img, threshold=0.2, scale=1.0)
            if len(bbox) == 0:
                print("found no faces on ", file)
                bbox, landmark = det_model.detect(img, threshold=0.05, scale=1.0)
            if len(bbox) == 0:
                embs = np.zeros(512)
                print(embs.shape)
                np.save(os.path.join(img_features_path, file), embs)
                continue
            if len(bbox) > 1:
                print("found multiple faces on ", file)
                bbox, landmark = det_model.detect(img, threshold=0.7, scale=1.0)
            bbox, landmarks = bbox[0], landmark[0]
            embs = (get_face_embeddings(img, landmarks, emb_model))[0]
            target_emb_list.append((embs))
            np.save(os.path.join(img_features_path, file), embs)
    return target_emb_list#os.path.join(output_path, 'img_features')