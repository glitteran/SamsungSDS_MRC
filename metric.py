#-*- coding: utf-8 -*-
import re
import string
import collections

from scipy.stats import pearsonr
from sklearn.metrics import f1_score, average_precision_score, accuracy_score
from seqeval.metrics import classification_report

import datasets


class KLUE(datasets.Metric):
    def _info(self):
        if self.config_name not in ["mrc"]:
            raise KeyError("No such dataset in KLUE")
        if self.config_name == "mrc":
            return datasets.MetricInfo(
                description="KLUE metric",
                citation="citation",
                inputs_description="predictions, references",
                features=datasets.Features(
                    {
                        "predictions": {
                            "guid": datasets.Value("string"),
                            "prediction_text": datasets.Value("string"),
                        },
                        "references": {
                            "guid": datasets.Value("string"),
                            "answers": datasets.features.Sequence(
                                {"text": datasets.Value("string"), "answer_start": datasets.Value("int32")}
                            ),
                        },
                    }
                ),
            )

    def _compute(self, predictions, references):
        if self.config_name == "mrc":
            dataset = [{"paragraphs": [{"qas": references}]}]
            predictions = dict((p["guid"], p["prediction_text"]) for p in predictions)

            exact_raw, f1_raw = get_raw_scores(dataset, predictions)
            
            total = len(exact_raw)
            return {
                "exact": 100.0 * sum(exact_raw.values()) / total,
                "f1": 100.0 * sum(f1_raw.values()) / total,
                "total": total
            }
        else:
            raise KeyError("No such dataset in KLUE")


def get_raw_scores(dataset, preds):
    exact_scores = {}
    f1_scores = {}
    for article in dataset:
        for p in article["paragraphs"]:
            for qa in p["qas"]:
                # 데이터에서 qid 값을 가져옴
                qid = qa["guid"]
                gold_answers = [t for t in qa["answers"]["text"] if normalize_answer(t)]
                if not gold_answers:
                    # For unanswerable questions, only correct answer is empty string
                    gold_answers = [""] # 정답이 존재하지 않는 경우 빈 문자열로 반환
                if qid not in preds: # 데이터에서 가져온 qid값이 모델이 예측한 guid에 없을 경우 넘김.  
                    print("Missing prediction for %s" % qid)
                    continue
                a_pred = preds[qid] # p["prediction_text"] 값을 가져옴. 
                # Take max over all gold answers
                exact_scores[qid] = max(compute_exact(a, a_pred) for a in gold_answers)
                f1_scores[qid] = max(compute_f1(a, a_pred) for a in gold_answers)
    return exact_scores, f1_scores

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        #띄어쓰기 단위로 쪼개서 다시 붙임. 
        return " ".join(text.split())

    def remove_punc(text):
        #string.punctuation: !"#$%&'()*+, -./:;<=>?@[\]^_`{|}~
        exclude = set(string.punctuation)
        #char하나하나 특수기호 다 제거하고 다 붙임
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        #소문자로 변환
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        ## 실제 정답값이 없거나, 모델이 정답이 없다고 예측할 경우 둘 다 같이 예측해야 1, 아니면 0으로 판단함. 
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def get_tokens(s):
    #띄어쓰기 단위로 쪼갬 
    if not s:
        return []
    return normalize_answer(s).split()