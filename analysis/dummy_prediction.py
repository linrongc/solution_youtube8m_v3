from data import Youtube8M
from util import parse_vid_preds
import heapq


class DummyPrediction:
    cache_folder = "cache/"

    def __init__(self, test_folder, base_folder=None, whitelist=None, limit=100000):
        self.test_folder = test_folder
        self.base_folder = base_folder or Youtube8M.fast_strat_segment_folder
        self.dataset = Youtube8M(self.test_folder, self.base_folder)
        self.whitelist = set(whitelist)
        self.limit = limit


class DummyVideoPrediction(DummyPrediction):
    def __init__(self, test_folder, base_folder=None, limit=100000, whitelist=None, topk=None):
        DummyPrediction.__init__(self, test_folder, base_folder, whitelist, limit)
        self.topk = topk

    def _generate_video_candidates(self, vid_preds):
        inv_index = {c: [] for c in self.whitelist}
        whitelist = set(self.whitelist)
        for vid, pred in vid_preds.iteritems():
            for c, score in pred.iteritems():
                if c in whitelist:
                    inv_index[c].append((vid, score))
        for c in inv_index.iterkeys():
            inv_index[c].sort(key=lambda x: x[1], reverse=True)
        return inv_index

    def dummy_video_preds(self, pred_path):
        print("dummy video preds from: ", pred_path)
        vid_preds = parse_vid_preds(pred_path)
        inv_index = self._generate_video_candidates(vid_preds)
        len_map = self.dataset.load_vid_len()
        results = {}
        for c in self.whitelist:
            count = 0
            pred = []
            for vid, score in inv_index[c]:
                vid_len = len_map[vid]
                vid_len = vid_len - vid_len % 5
                for n in range(0, vid_len, 5):
                    pred.append(("{}:{}".format(vid, n), score))
                    count += 1
                    if self.limit and count >= self.limit:
                        break
                if self.limit and count >= self.limit:
                    break
            results[c] = pred
        return results


class VidFilter:
    def __init__(self, vid_preds):
        self.vid_preds = vid_preds

    def filter(self, seg_id, pred):
        vid = seg_id.split(":")[0]
        if vid in self.vid_preds:
            vid_pred = self.vid_preds[vid]
            return {key: value for key, value in pred.iteritems() if key in vid_pred}
        else:
            return {}


class DummySegmentPrediction(DummyPrediction):
    def __init__(self, test_folder, base_folder=None, whitelist=None, limit=100000,
                 vid_filter=None, seg_filter=None):
        DummyPrediction.__init__(self, test_folder, base_folder, whitelist, limit)
        self.vid_filter = vid_filter
        self.seg_filter = seg_filter

    @classmethod
    def seg_preds_gtr(cls, pred_path):
        with open(pred_path, 'r') as f:
            f.readline()
            for line in f:
                fields = line.rstrip("\n").split(",")
                seg_id = fields[0]
                pred_str = fields[1].split()
                pred = {int(l): float(s) for l, s in zip(pred_str[::2], pred_str[1::2])}
                yield seg_id, pred

    def filter_pred(self, seg_id, pred):
        if self.vid_filter is not None:
            pred = self.vid_filter.filter(seg_id, pred)
        if self.seg_filter is not None:
            pred = self.seg_filter.filter(seg_id, pred)
        return pred

    def dummy_seg_preds(self, pred_path):
        count = 0
        results = {c: [] for c in self.whitelist}
        for seg_id, pred in self.seg_preds_gtr(pred_path):
            pred = self.filter_pred(seg_id, pred)
            for c, seg_score in pred.iteritems():
                if c in self.whitelist:
                    results[c].append((seg_id, seg_score))
            count += 1
            if count % 10000 == 0:
                print("process {} segments.\n".format(count))

        for c in results.iterkeys():
            results[c] = sorted(results[c], key=lambda x: x[1], reverse=True)[:self.limit]
        return results

    def dummy_seg_preds_light(self, pred_path):
        count = 0
        heaps = {c: [] for c in self.whitelist}
        for seg_id, pred in self.seg_preds_gtr(pred_path):
            pred = self.filter_pred(seg_id, pred)
            for c, seg_score in pred.iteritems():
                if c in self.whitelist:
                    if len(heaps[c]) >= self.limit:
                        heapq.heappushpop(heaps[c], (seg_score, seg_id))
                    else:
                        heapq.heappush(heaps[c], (seg_score, seg_id))
            count += 1
            if count % 10000 == 0:
                print("process {} segments.\n".format(count))
        for cls, cls_heap in heaps.iteritems():
            heaps[cls] = [(x[1], x[0]) for x in sorted(cls_heap, key=lambda x: x[0], reverse=True)]
        return heaps


if __name__ == "__main__":
    from util import load_whitelist_topics
    import os, json
    # video_pred_path = "../results/mix3_block_nextvlad_1T_8g_5l2_5drop_128k_2048_2x80_logistic.csv"
    # segment_pred_path = "../results/nextvlad_16g_5l2_5drop_128k_1024_2x80_logistic_5f_validate.csv"
    video_pred_path = "../results/double_reverse_nextvlad_8g_5l2_5drop_128k_2048_2x80_logistic_strat_test.csv"
    segment_pred_path = "../results/mix3_nextvlad_1T1_8g_5l2_5drop_128k_2048_2x80_logistic_final_5f_10ep_3T1_0div_75drop_pretrain_validate_train_strat_test_k1000.csv"
    whitelist = load_whitelist_topics()
    print("whitelist: ", len(whitelist))

    cache_path = "result_cache/mix3_v2_combine_v01_s09_rtop20_pretrain_v2_light_3T1_final.json"
    if not os.path.exists(cache_path):
        dummy_vid_pred = DummyVideoPrediction("validate_strat_split/test", base_folder=Youtube8M.dream_segment_folder, whitelist=whitelist)
        # results = dummy_vid_pred.dummy_video_preds(video_pred_path)

        vid_preds = parse_vid_preds(video_pred_path)
        vid_filter = VidFilter(vid_preds)
        dummy_seg = DummySegmentPrediction("validate_strat_split/test", base_folder=Youtube8M.dream_segment_folder, whitelist=whitelist, vid_filter=vid_filter)
        results = dummy_seg.dummy_seg_preds_light(segment_pred_path)
        with open(cache_path, 'w') as f:
            json.dump(results, f)
    else:
        with open(cache_path, 'r') as f:
            results = json.load(f)
            results = {int(x):y for x, y in results.iteritems()}
    from metrics import MAPK, Recall, Precision
    dataset = Youtube8M("validate_strat_split/test", Youtube8M.dream_segment_folder)
    labels = dataset.load_label_map()
    mapk = MAPK(results, labels, pred_has_score=True)
    print(mapk.score_per_class())
    print(mapk.mean_class_score())

    recall = Recall(results, labels, pred_has_score=True)
    print(recall.score_per_class())
    print(recall.mean_class_score())

    precision = Precision(results, labels, pred_has_score=True)
    print(precision.score_per_class())
    print("precision: ", precision.mean_class_score())