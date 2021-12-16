import os
import sys
import detection
import recognition
import cluster_association
import input_association
import create_output
import association_utils

if __name__ == '__main__':

    #get video path
    dir_path = os.path.dirname(os.path.realpath(__file__))
    video_folder_path = os.path.join(dir_path, "Video")
    video_path = os.path.join(video_folder_path, os.listdir(video_folder_path)[0])

    #determine association method
    print('Video to be pixelated should be located in: ', os.path.join(dir_path, 'Video'))
    association_method = input('Enter association method: (1: Image Input, 2: Clustering).')
    print('association_method: ', association_method)

    #create folder to store outputs
    working_dir = os.path.join(dir_path, 'output')
    if not os.path.exists(working_dir):
        os.mkdir(working_dir)
    f_name, f_ext = os.path.splitext(os.path.basename(video_path))
    output_dir = os.path.join(working_dir, f_name)
    os.mkdir(output_dir)
    meta_path = os.path.join(output_dir, 'metadata')
    os.mkdir(meta_path)

    #association methods: clustering, input association
    #clustering requires an estimation of the number of all appearing faces in the Video. The user picks the persons to be anonymized by sample images taken from the Video.
    #input association requires sample images provided by the user of faces to be anonymized

    #input association
    if association_method == '1':

        change_sim = input("Change similarity threshold?(y/n) (Default: 0.3)")
        if change_sim == "y":
            sim_threshold = float(input("Enter a similarity threshold between 0 and 1: \n"
                               "(A low threshold value will increase rate of pixelation but also number of falsely pixelated faces. For a high threshold the opposite applies.)"))
        else:
            sim_threshold = 0.3
        print("Sample images of persons to be pixelated should be located in: " + os.path.join(dir_path, 'input_imgs')
              + "(Images of each face from multiple angles may lead to better results.)")
        proceed = input("Proceed? (y/n)")
        if proceed == "n":
            sys.exit()

        # create detections file
        det_file = detection.create_det_file(video_path, meta_path)
        print("created detection file")

        # extract embeddings and link single detections into continuous tracks
        tracks_path = recognition.create_tracks_and_emb_file(video_path, meta_path)
        print("created tracks and embeddings files")

        sample_img_path = os.path.join(dir_path, 'input_imgs')
        print(sample_img_path)
        target_emb_list = association_utils.create_npy_from_imgs(sample_img_path, meta_path)
        labels = input_association.associate_by_input_no_skip(target_emb_list, meta_path, sim_threshold)
        print('labels', labels)
        pix_vid_mute = create_output.pixelate_tracks_input(video_path, tracks_path, labels)
        create_output.add_audio(pix_vid_mute, video_path, output_dir)

    #clustering
    elif association_method == '2':
        cluster_nr = int(input("Enter estimated number of persons in video: \n"
                           "(Choosing higher numbers requires more manual associating, but will improve accuracy.)"))

        # create detections file
        det_file = detection.create_det_file(video_path, meta_path)
        print("created detection file")

        # extract embeddings and link single detections into continuous tracks
        tracks_path = recognition.create_tracks_and_emb_file(video_path, meta_path)
        print("created tracks and embeddings files")

        #group tracks into clusters based on feature similarity
        labels, avg_embs_list = cluster_association.HAC(meta_path, tracks_path, cluster_nr)

        #store sample images of each cluster
        cluster_association.update_sample_imgs_clusters_new(meta_path, avg_embs_list, labels, cluster_nr)

        #get target ids by user input
        target_ids = input("Enter IDs to be pixelated from \"" + os.path.join(output_dir, "sample_imgs") + "\":  (Example: 1, 5, 14) \n").split(",")
        target_ids = [int(id) for id in target_ids]
        tracks_dict = cluster_association.create_dict_for_labels(labels, meta_path)

        #assign detections on each frame their target id
        det_with_ids = cluster_association.create_det_with_ids(tracks_dict, tracks_path)

        #create pixelated output video using the updated detections file
        pix_vid_mute = create_output.pixelate_tracks_cluster(video_path, det_with_ids, target_ids)

        #add the original audio to the video output
        create_output.add_audio(pix_vid_mute, video_path, output_dir)

    print("Finished pixelation. New Video is located in: " + output_dir)
