from data import Youtube8M
from dummy_prediction import DummyVideoPrediction, DummySegmentPrediction
from util import load_whitelist_topics


class ValueModel:

    def __init__(self, video_ranks, whitelist,
                 video_exp=0.2, seg_exp=0.8, cutoff=0.,
                 value_func="exp_merge", rerank=10000
        ):
        self.video_exp = video_exp
        self.seg_exp = seg_exp

        self.video_preds = self.truncate_map(video_ranks, rerank)

        self.whitelist = set(whitelist)
        self.cutoff = cutoff
        self.value_func = value_func
        self.results = {}

    @classmethod
    def truncate_map(cls, video_ranks, rerank):
        results = {}
        for c, preds in video_ranks.iteritems():
            results[c] = dict(preds[:rerank])
        return results

    def exp_merge(self, video_score, seg_score):
        return (video_score ** self.video_exp) * (seg_score ** self.seg_exp)

    def _generate_pred(self, seg_preds):
        value_func = getattr(self, self.value_func)
        count= 0
        for seg_id, pred in seg_preds:
            count += 1
            if count % 10000 == 0:
                print("process {} segments.\n".format(count))
            for c, seg_score in pred.iteritems():
                if c in self.whitelist:
                    vid_score = self.video_preds.get(c, {}).get(seg_id, 0)
                    value = value_func(vid_score, seg_score)
                    if value > self.cutoff:
                        yield seg_id, c, value


    def retrieval(self, seg_preds):
        self.results = {}
        for seg_id, c, value in self._generate_pred(seg_preds):
            if c in self.results:
                self.results[c].append((seg_id, value))
            else:
                self.results[c] = [(seg_id, value),]

        for c in self.results.iterkeys():
            self.results[c].sort(key=lambda x: x[1], reverse=True)
        return self.results

def add_fallbacks(pred_map, fallback_map):
    print("adding fallbacks...")
    for c in pred_map.iterkeys():
        covered = set([x[0] for x in pred_map[c]])
        fallback = fallback_map[c]
        total = len(pred_map[c])
        for seg, score in fallback:
            if total >= 100000:
                break
            if seg not in covered:
                pred_map[c].append((seg, score))
                total += 1

def generate_pred_file(limit=100000):

    vid_pred_path = "../results/parallel4_mix3_nextvlad_x1_1T1_8g_5l2_5drop_256k_2048_120_logistic_fix_0div_final_v3_test_top20.csv"
    seg_pred_path = "../results/parallel4_mix3_nextvlad_x2_1T1_8g_5l2_5drop_128k_2048_80_logistic_fix_0div_final_711k_5f_10ep_20T400_20T400_0div_75drop_512_pretrain_validate_train_4l2_test_single_k1000.csv"
    fallback_vid_pred_path = "../results/mix3_nextvlad_1T1_8g_5l2_5drop_128k_2048_2x80_logistic_fix_0div_final_test_k1000.csv"
    output = "../results/mix12_final_v005_s095_rtop20_nextvlad_distill.csv"

    whitelist = load_whitelist_topics("../segment_label_ids.csv")

    dummy_video = DummyVideoPrediction("test", Youtube8M.dream_segment_folder, whitelist=whitelist)
    video_ranks = dummy_video.dummy_video_preds(vid_pred_path)

    vmodel = ValueModel(video_ranks=video_ranks, whitelist=whitelist, video_exp=0.05, seg_exp=0.95, rerank=None)

    pred_map = vmodel.retrieval(DummySegmentPrediction.seg_preds_gtr(seg_pred_path))

    fallback_map = dummy_video.dummy_video_preds(fallback_vid_pred_path)
    add_fallbacks(pred_map, fallback_map)

    write_preds_map(pred_map, output, whitelist, limit)

def write_preds_map(preds_map, out_path, whitelist, limit=100000):
    with open(out_path, 'w') as f:
        f.write("Class,Segments\n")
        for c in whitelist:
            c_preds = preds_map[c]
            count = 0
            f.write(str(c) + ",")
            results = []
            for seg_id, score in c_preds:
                count += 1
                results.append(seg_id)
                if count >= limit:
                    break
            print(c, count)
            f.write(" ".join(results) + "\n")

if __name__ == "__main__":
    generate_pred_file()


