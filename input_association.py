
import os
import numpy as np
from utils import compute_sim

#from testing import metrics
def associate_by_input(target_emb_list, output_path, sim_threshold):
    embs_path = os.path.join(output_path, 'track_embs')
    track_embs_list = []
    for file in os.listdir(embs_path):
        if '.npy' in file:
            embs = np.load(os.path.join(embs_path, file))
            embs_avg = embs.sum(axis=0) / len(embs)
            track_embs_list.append(embs_avg)
    track_labels = np.array([])

    for track_emb in track_embs_list:
        #-1 means no match
        highest_match = -1
        highest_sim = 0
        for id, input_emb in enumerate(target_emb_list):
            sim = compute_sim(track_emb, input_emb)
            print('sim: ',sim)
            if sim > sim_threshold and sim > highest_sim:
                highest_match = id
                highest_sim = sim
        track_labels = np.append(track_labels, highest_match)
        print('track_labels', track_labels)
    return track_labels


def associate_by_input_no_skip(input_embs_list, output_path, sim_threshold):
    embs_path = os.path.join(output_path, 'track_embs')
    track_id_list = []
    track_embs_list = []
    for file in os.listdir(embs_path):
        if '.npy' in file:
            embs = np.load(os.path.join(embs_path, file))
            embs_avg = embs.sum(axis=0) / len(embs)
            track_embs_list.append(embs_avg)
            f_name, f_ext = os.path.splitext(os.path.basename(file))
            if f_name != '000000':
                f_name = f_name.lstrip('0')
            else:
                f_name = '0'
            f_name = int(f_name)
            track_id_list.append(f_name)
    track_labels = np.array([])

    track_embs_list_index = 0
    for index in range(max(track_id_list) + 1):
        if index not in track_id_list:
            track_labels = np.append(track_labels, -1)
        else:
            track_emb = track_embs_list[track_embs_list_index]
            #-1 means no match
            highest_match = -1
            highest_sim = 0
            for id, input_emb in enumerate(input_embs_list):
                sim = compute_sim(track_emb, input_emb)
                if sim > sim_threshold and sim > highest_sim:
                    highest_match = id
                    highest_sim = sim
            track_labels = np.append(track_labels, highest_match)
            track_embs_list_index += 1
    print('track_labels', track_labels)
    return track_labels


def set_labels_from_selection(selected_clusters, labels):
    new_labels = []
    for label in labels:
        if label in selected_clusters:
            new_labels.append(label)
        else:
            new_labels.append(-1)
    return new_labels

def set_labels_from_selection2(selected_clusters, labels):
    selected_clusters = [val for sublist in selected_clusters for val in sublist]
    new_labels = []
    for label in labels:
        if label in selected_clusters:
            new_labels.append(label)
        else:
            new_labels.append(-1)
    return new_labels