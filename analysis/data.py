import tensorflow as tf
import os
import json


class Youtube8M:
    dream_segment_folder = "/media/linrongc/dream/data/yt8m/frame/3/"
    dream_frame_folder = "/media/linrongc/dream/data/yt8m/frame/2/"
    dream_strat_segment_folder = dream_segment_folder + "validate_strat_split/"

    fast_segment_folder = "/media/linrongc/fast/data/yt8m/frame/3/"
    fast_strat_segment_folder = fast_segment_folder + "validate_strat_split/"

    cache_folder = "cache/"

    def __init__(self, test_folder, base_folder=None):
        self.test_folder = test_folder
        self.base_folder = base_folder or self.fast_segment_folder

    @classmethod
    def parse_video(cls, example, seg_labels=False):
        tf_example = tf.train.Example.FromString(example)
        vid_id = tf_example.features.feature['id'].bytes_list.value[0].decode(encoding='UTF-8')
        labels = tf_example.features.feature['labels'].int64_list.value
        data = {"id": vid_id, "labels": labels}
        if seg_labels:
            data["segment_labels"] = tf_example.features.feature["segment_labels"].int64_list.value
            data["segment_start_times"] = tf_example.features.feature["segment_start_times"].int64_list.value
            data["segment_scores"] = tf_example.features.feature["segment_scores"].float_list.value
        return data

    @classmethod
    def parse_frame(cls, example):
        tf_seq_example = tf.train.SequenceExample.FromString(example)
        n_frames = len(tf_seq_example.feature_lists.feature_list['audio'].feature)

        rgb_frames = []
        audio_frame = []
        # iterate through frames

        '''
        for i in range(n_frames):
            rgb_frames.append(tf.cast(tf.decode_raw(
                tf_seq_example.feature_lists.feature_list['rgb']
                    .feature[i].bytes_list.value[0], tf.uint8)
                , tf.float32).eval())
            audio_frame.append(tf.cast(tf.decode_raw(
                tf_seq_example.feature_lists.feature_list['audio']
                    .feature[i].bytes_list.value[0], tf.uint8)
                , tf.float32).eval())
        '''
        return {
            "frame_num": n_frames,
            "rgb_frames": rgb_frames,
            "audio_frames": audio_frame,
        }

    @classmethod
    def generate_videos(cls, path, video_data=True, frame_data=False, segment_labels=False):
        files = os.listdir(path)
        tf_records = [x for x in files if x.endswith("tfrecord")]
        if frame_data:
            sess = tf.InteractiveSession()
        print("reading records in ", path)
        count = 0
        for record in tf_records:
            record_path = os.path.join(path, record)
            for example in tf.python_io.tf_record_iterator(record_path):
                data = {}
                if video_data:
                    data["video_level"] = cls.parse_video(example, segment_labels)
                if frame_data:
                    data["frame_level"] = cls.parse_frame(example)
                count += 1
                if count % 10000 == 0:
                    print(count)
                yield data
        if frame_data:
            sess.close()

    def load_validate_seg_labels(self):
        cache_path = os.path.join(self.cache_folder, self.test_folder.replace("/", "_") + "_label.json")
        if not os.path.exists(cache_path):
            seg_label_map = {}
            for video_data in self.generate_videos(os.path.join(self.base_folder, self.test_folder), frame_data=False, segment_labels=True):
                vid_id = video_data['video_level']['id']
                seg_labels = video_data['video_level']['segment_labels']
                seg_scores = video_data['video_level']['segment_scores']
                seg_start_times = video_data['video_level']['segment_start_times']
                seg_label_map[vid_id] = [
                    (label, score, start_time)
                    for label, score, start_time in zip(seg_labels, seg_scores, seg_start_times)
                ]
            with open(cache_path, 'w') as f:
                json.dump(seg_label_map, f)
        else:
            print("load cache from {}".format(cache_path))
            with open(cache_path, 'r') as f:
                seg_label_map = json.load(f)
        return seg_label_map

    def load_vid_len(self):
        cache_path = os.path.join(self.cache_folder, self.test_folder.replace("/", "_") + "_vid_len.json")
        if not os.path.exists(cache_path):
            vid_len_map = {}
            for video_data in self.generate_videos(os.path.join(self.base_folder, self.test_folder), frame_data=True,
                                                        segment_labels=False):
                vid = video_data['video_level']['id']
                num_frames = video_data['frame_level']['frame_num']
                vid_len_map[vid] = num_frames
            with open(cache_path, 'w') as f:
                json.dump(vid_len_map, f)
        else:
            with open(cache_path, "r") as f:
                vid_len_map = json.load(f)
        print("finish loading video length...")
        return vid_len_map

    def load_label_map(self):
        label_map = {}
        vid_seg_labels = self.load_validate_seg_labels()
        for vid, seg_labels in vid_seg_labels.iteritems():
            for label, score, start_time in seg_labels:
                id = "{}:{}".format(vid, start_time)
                if label in label_map:
                    label_map[label].append((id, score))
                else:
                    label_map[label] = [(id, score)]
        return label_map


if __name__ == "__main__":
    dataset = Youtube8M("test", base_folder=Youtube8M.fast_segment_folder)
    # seg_labels = dataset.load_validate_seg_labels()
    vid_lens = dataset.load_vid_len()
    # print(len(seg_labels))
    print(len(vid_lens))