def parse_line(line, norm_score=False, cutoff=-1):
    fields = line.split(",")
    vid = fields [0]
    pred_str = fields[1].split()
    pred = {int(l): float(s) for l, s in zip(pred_str[::2], pred_str[1::2])}
    if norm_score:
        total = sum(pred.itervalues())
        pred = {key: value / total for key, value in pred.iteritems()}
    if cutoff!=-1:
        threshold = sorted(pred.itervalues())[cutoff]
        pred = {key: value for key, value in pred.iteritems() if value >= threshold}
    return [vid, pred]

def weighted_log_sum(preds, weights, out_path, normalize_score=False, cutoff=-1):
    pred_f = [open(f, 'r') for f in preds]
    for f in pred_f:  # skip header
        f.readline()
    count = 0
    with open(out_path, "w") as outf:
        outf.write("VideoId,LabelConfidencePairs\n")
        while True:
            input_str = [f.readline() for f in pred_f]
            eof = False
            for pred_line in input_str:
                if not pred_line or len(pred_line) == 0:
                    eof = True
                    break
            if eof:
                print(input_str)
                break
            preds = [parse_line(line, normalize_score, cutoff=cutoff) for line in input_str]
            out_pred = {}
            vid = preds[0][0]
            weight_map = {}
            for pred, weight in zip(preds, weights):
                pred_vid = pred[0]
                if pred_vid != vid:
                    print("vid mismatch!")
                for c, score in pred[1].iteritems():
                    if c not in out_pred:
                        out_pred[c] = score ** weight
                    else:
                        out_pred[c] *= score ** weight
                    weight_map[c] = weight_map.get(c, 0) + weight
            weight_sum = sum(weights)
            out_pred_list = sorted(
                [(key,value) for (key, value) in out_pred.iteritems() if weight_map[key] == weight_sum],
                key=lambda x: x[1], reverse=True
            )
            outf.write(vid + ",")
            outf.write(" ".join(["{} {}".format(key, value) for key, value in out_pred_list]))
            outf.write("\n")
            count += 1
            if count % 10000 == 0:
                print("process %d lines" % count)
    print(count)

def merge_seg_preds():
    preds = [
        "../results/parallel4_mix3_nextvlad_x2_1T1_8g_5l2_5drop_256k_1024_64_logistic_fix_0div_final_711k_5f_10ep_20T400_20T400_0div_75drop_512_pretrain_validate_train_4l2_test_k1000.csv",
        "../results/parallel4_mix3_nextvlad_x1_1T1_8g_5l2_5drop_256k_2048_120_logistic_fix_0div_final_574140_5f_10ep_20T400_20T400_0div_75drop_512_pretrain_validate_train_4l2_test_k1000.csv",
        "/media/linrongc/2019/result_archive/parallel4_mix3_nextvlad_x2_1T1_8g_5l2_5drop_128k_2048_120_logistic_fix_0div_final_710567_5f_10ep_50T2500_0div_75drop_512_pretrain_validate_train_4l2_test_k1000.csv",
    ]
    weights = [0.3, 0.3, 0.4]
    out_path = "../results/mix12_mix4_nextvlad_combine_1000k_cut500.csv"
    weighted_log_sum(preds, weights, out_path, normalize_score=False, cutoff=500)


if __name__ == "__main__":
    merge_seg_preds()