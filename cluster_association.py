import os
import csv
import cv2
from sklearn.cluster import AgglomerativeClustering
from numpy.linalg import norm
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances


# method to calculate distance between two samples
import association_utils


def compute_dissimilarity(emb1, emb2):
    emb1 = emb1.flatten()
    emb2 = emb2.flatten()
    sim = np.dot(emb1, emb2) / (norm(emb1) * norm(emb2))
    return -1 * sim


# method to calculate distance between two samples
def compute_sim(emb1, emb2):
    emb1 = emb1.flatten()
    emb2 = emb2.flatten()
    sim = np.dot(emb1, emb2) / (norm(emb1) * norm(emb2))
    return sim


# Method to calculate distances between all sample pairs

def sim_affinity(X):
    return pairwise_distances(X, metric=compute_dissimilarity)


def k_means(embs_path, nr_of_clusters):
    emb_list = []
    track_list = []
    for file in os.listdir(embs_path):
        if '.npy' in file:
            f_name, f_ext = os.path.splitext(os.path.basename(file))
            embs = np.load(embs_path + file)
            embs_avg = embs.sum(axis=0) / len(embs)
            emb_list.append(embs_avg)
            track_list.append(f_name)
    # average embeddings list
    emb_list = np.array(emb_list)
    #    print("sim: ", compute_sim(emb_list[0], emb_list[1]))

    kmeans = KMeans(n_clusters=nr_of_clusters)
    labels = kmeans.fit_predict(emb_list)
    return labels, emb_list, track_list


def HAC(output_dir, tracks_path, nr_of_clusters, dist_metric='sim_affinity'):
    embs_path = os.path.join(output_dir, 'track_embs')
    avg_emb_list = []
    track_list = []
    for file in os.listdir(embs_path):
        if '.npy' in file:
            f_name, f_ext = os.path.splitext(os.path.basename(file))
            embs = np.load(os.path.join(embs_path, file))
            embs_avg = embs.sum(axis=0) / len(embs)
            avg_emb_list.append(embs_avg)
            track_list.append(f_name)
    # average embeddings list
    avg_emb_list = np.array(avg_emb_list)

    # Creating the model
    if dist_metric == "sim_affinity":
        agg_clustering = AgglomerativeClustering(n_clusters=nr_of_clusters, affinity=sim_affinity,
                                                 linkage='average')  # "euclidean", ward
    if dist_metric == "euclidean":
        agg_clustering = AgglomerativeClustering(n_clusters=nr_of_clusters, affinity="euclidean",
                                                 linkage='ward')  # "euclidean", ward
    # predicting the labels
    labels = agg_clustering.fit_predict(avg_emb_list)
    print(labels)
#    create_det_with_ids(labels, tracks_path)
    return labels, avg_emb_list



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
    return new_csv_path
def update_sample_imgs_clusters_new(meta_path, avg_embs_list, track_labels, cluster_nr):
    output_path = os.path.dirname(meta_path) + "/sample_imgs/"
    track_img_path = os.path.join(meta_path, 'track_imgs')
    track_emb_path = os.path.join(meta_path, 'track_embs')
    img_embs = np.load(os.path.join(meta_path, 'img_features.npy'))
    os.mkdir(output_path)
    final_img_list = []
    #list of width for each track image
    img_size_list = []
    track_name_list = []
    for npy in os.listdir(track_emb_path):
        f_name, f_ext = os.path.splitext(os.path.basename(npy))
        img_path = os.path.join(track_img_path, f_name + ".jpg")
        track_name_list.append(f_name)
        if os.path.exists(img_path):
            img_temp = cv2.imread(img_path)
            h, w, _ = img_temp.shape
            img_size_list.append(w)
        else:
            img_size_list.append(0)
    for i in range(cluster_nr):
        #get index of all tracks that belong to cluster i
        track_ids = [index for index, x in enumerate(track_labels) if x == i]
        #get emb_avg for all tracks that belong to cluster i
        track_avg_list = [emb for index, emb in enumerate(avg_embs_list) if index in track_ids]
        cluster_embs_avg = sum(track_avg_list) / len(track_avg_list)
        #sample_score = sim * width (instead of height, so frontal faces are preferred)
        sample_score_list = []
        for index, id in enumerate(track_ids):
            if id+1 > len(img_embs):
                sample_score_list.append(0)
                continue
            sim = compute_sim(img_embs[id], cluster_embs_avg)
            width = img_size_list[track_ids[index]]
            sample_score_list.append(sim*width)
        highest_score = max(sample_score_list)
        high_score_index = [i for i, j in enumerate(sample_score_list) if j == highest_score][0]
        high_score_id = track_name_list[track_ids[high_score_index]]
        final_img_list.append(str(high_score_id) + '.jpg')
    for index, img_name in enumerate(os.listdir(track_img_path)):
        if img_name in final_img_list:
            img = cv2.imread(os.path.join(track_img_path, img_name))
            cv2.imwrite(output_path + "person" + str(final_img_list.index(img_name)) + '.jpg', img)

def create_dict_for_labels(track_labels, results_path):
    # create dictionary mapping for track id name -> label
    embs_path = os.path.join(results_path, "track_embs/")
    track_id_list = []
    for file in os.listdir(embs_path):
        print('read file')
        if '.npy' in file:
            f_name, f_ext = os.path.splitext(os.path.basename(file))
            if f_name != '000000':
                f_name = f_name.lstrip('0')
            else:
                f_name = '0'
            track_id_list.append(f_name)
    track_dict = {}  # Empty dictionary to add values into

    for index, f_name in enumerate(track_id_list):
        track_dict[f_name] = track_labels[index]
    return track_dict
