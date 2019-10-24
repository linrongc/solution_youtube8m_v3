from sklearn import metrics

class BasicMetric:

    def __init__(self, predictions_map, labels_map, pred_has_score=False):
        self.predictions_map = predictions_map
        self.labels_map = labels_map
        self.pred_has_score = pred_has_score

    def intersect(self, preds, labels):
        pred_keys = set([x[0] for x in preds])
        label_keys = set([x[0] for x in labels])
        common_keys = pred_keys.intersection(label_keys)
        if len(common_keys) == 0:
            return None, None
        label_dict = dict(labels)


        common_preds = []
        common_labels = []

        for pred in preds:
            if pred[0] in common_keys:
                common_preds.append(pred)
                common_labels.append((pred[0], label_dict[pred[0]]))

        return common_preds, common_labels

    def calculate_scores(self, preds, labels):
        raise NotImplementedError()

    def score_per_class(self, default_value=0.):
        scores = {}
        for c, labels in self.labels_map.iteritems():
            if c not in self.predictions_map:
                scores[c] = default_value
            else:
                scores[c] = self.calculate_scores(self.predictions_map[c], labels)
        return scores

    def mean_class_score(self, default_value=0.):
        scores = self.score_per_class(default_value=default_value)
        return sum(scores.itervalues()) / len(scores)



class AUC(BasicMetric):

    def __init__(self, predictions_map, labels_map, pred_has_score):
        if not pred_has_score:
            print("Need prediction scores!")
        BasicMetric.__init__(self, predictions_map, labels_map)

    def calculate_scores(self, raw_preds, raw_labels):
        preds, labels = self.intersect(raw_preds, raw_labels)

        if not preds or not labels:
            return 0.

        fpr, tpr, thresholds = metrics.roc_curve(
            [x[1] for x in labels],
            [x[1] for x in preds]
        )
        return metrics.auc(fpr, tpr)


class MAPK(BasicMetric):

    def __init__(self, predictions_map, labels_map, k=100000, pred_has_score=False):
        BasicMetric.__init__(self, predictions_map, labels_map, pred_has_score)
        self.k = k

        
    def calculate_scores(self, raw_preds, raw_labels):
        pos_count = len([x for x in raw_labels if x[1] > 0])
        raw_label_map = dict(raw_labels)
        if self.pred_has_score:
            preds = [x[0] for x in sorted(raw_preds, key=lambda x: x[1], reverse=True)]
        else:
            preds = raw_preds
        pos = 0.
        neg = 0.
        score = 0.
        for seg_id in preds[:self.k]:
            if seg_id in raw_label_map:
                if raw_label_map[seg_id] > 0:
                    pos += 1
                    score += pos / (pos + neg)
                else:
                    neg += 1
        return score / pos_count


class PR(BasicMetric):
    pass

class Recall(BasicMetric):
    def __init__(self, predictions_map, labels_map, pred_has_score=False):
        BasicMetric.__init__(self, predictions_map, labels_map, pred_has_score)

    def calculate_scores(self, raw_preds, raw_labels):
        if self.pred_has_score:
            preds = [x[0] for x in sorted(raw_preds, key=lambda x: x[1], reverse=True)]
        else:
            preds = raw_preds

        raw_label_map = dict(raw_labels)
        pos_label_count = len([x for x in raw_labels if x[1] > 0])
        hit_count = len([x for x in preds if raw_label_map.get(x, 0)> 0])
        return hit_count / float(pos_label_count) if pos_label_count > 0. else 0.


class Precision(BasicMetric):
    def __init__(self, predictions_map, labels_map, pred_has_score=False):
        BasicMetric.__init__(self, predictions_map, labels_map, pred_has_score)

    def calculate_scores(self, raw_preds, raw_labels):
        if self.pred_has_score:
            preds = [x[0] for x in sorted(raw_preds, key=lambda x: x[1], reverse=True)]
        else:
            preds = raw_preds
        raw_label_map = dict(raw_labels)
        hit_count = len([x for x in preds if raw_label_map.get(x, 0) > 0])
        overlap_count = len([x for x in preds if x in raw_label_map])
        return hit_count / float(overlap_count) if overlap_count > 0 else 0.


def unit_test():
    from data import Youtube8M
    from util import load_prediction_map
    preds_map = load_prediction_map("../results/dummy.csv")
    dataset = Youtube8M(test_folder="validate_strat_split/test", base_folder=Youtube8M.fast_segment_folder)
    labels_map = dataset.load_label_map()

    mapk = MAPK(preds_map, labels_map, pred_has_score=False)
    print(mapk.score_per_class())
    print(mapk.mean_class_score())

    recall = Recall(preds_map, labels_map, pred_has_score=False)
    print(recall.score_per_class())
    print(recall.mean_class_score())

    precision = Precision(preds_map, labels_map, pred_has_score=False)
    print(precision.score_per_class())
    print("precision: ", precision.mean_class_score())

    # metric = AUC(preds_map, labels_map)
    # print(metric.score_per_class())
    # print(metric.mean_class_score())


if __name__ == "__main__":
    unit_test()