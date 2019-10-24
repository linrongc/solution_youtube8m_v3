
def load_whitelist_topics(path="../segment_label_ids.csv"):
    with open(path, 'r') as f:
        return [int(x.rstrip("\n")) for x in f.readlines()[1:]]


def load_prediction_map(path, skip=1):
    pred_map = {}
    with open(path, 'r') as f:
        for _ in range(skip):
            f.readline()
        for line in f:
            fields = line.rstrip("\n").rstrip().split(",")
            label = int(fields[0])
            preds = fields[1].split()
            pred_map[label] = preds
    return pred_map

def dump_prediction_map(path):
    with open(path, 'w') as f:
        f.write("class,")


def parse_vid_preds(path):
    print('parsing video prediction..')
    vid_preds = {}
    with open(path, 'r') as f:
        f.readline()
        for line in f:
            fields = line.rstrip("\n").split(",")
            vid_id = fields[0]
            pred_str = fields[1].split()
            pred = {int(l): float(s) for l, s in zip(pred_str[::2], pred_str[1::2])}
            vid_preds[vid_id] = pred
    return vid_preds
