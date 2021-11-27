import utils
class Track:
    def __init__(self, id, embeddings, sample_img, last_seen_bbox, last_seen_frame):
        self.id = id
        self.tracklet_list = []
        self.emb_list = [embeddings]
        self.avg_emb = embeddings
        self.embeddings_counter = 1
        # list of indexes of csv file where person is identified
        self.detection_list = []
        #self.sample_img_index = 0
        self.last_seen_bbox = last_seen_bbox
        self.last_seen_frame = last_seen_frame
        self.update_counter = 0

        self.sample_img = sample_img
        # similarity of sample img and avg_emb
        self.sample_img_score = 0
        self.sample_img_emb = None
        # self.original_emb = original_emb
    def update_sample_img(self, new_img, new_emb, landmarks):
        h, w, c = new_img.shape
        new_score = w
        sim = utils.compute_sim(new_emb, self.avg_emb)
        if new_score > self.sample_img_score and sim > 0.3:
            if(len(self.emb_list) > 1):
                print("updating")
            self.sample_img = new_img
            self.sample_img_score = new_score
            self.sample_img_emb = new_emb
    def update_sample_img_old(self, new_img, new_emb, landmarks):
        h, w, c = new_img.shape
        #score is similarity * avg of width and height
        new_score = utils.compute_sim(new_emb, self.avg_emb) * ((h + w)/2)
        if new_score > self.sample_img_score:
            self.sample_img = new_img
            self.sample_img_score = new_score
    def update_embeddings_avg(self, new_embeddings):
        self.avg_emb = (self.avg_emb * self.embeddings_counter/(self.embeddings_counter + 1)) + (new_embeddings * 1/(self.embeddings_counter + 1))
        self.embeddings_counter += 1

    def update_embeddings_list(self, new_embeddings):
        self.emb_list.append(new_embeddings)