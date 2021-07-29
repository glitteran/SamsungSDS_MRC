import os
import json
import random
import argparse
import collections
import numpy as np
import torch
from typing import Optional, Tuple
from tqdm.auto import tqdm


def set_seed(seed: int):
    """
    random 이나 torch에서 사용되는 seed를 고정해야 
    같은 seed에 대해 항상 같은 결과를 도출 할 수 있음. 
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # Multi GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    # ^^ safe to call this function even if cuda is not available

def postprocess_qa_predictions(
    examples,#전처리 전 eval 데이터
    features,#전처리 후 eval 데이터 
    predictions: Tuple[np.ndarray, np.ndarray],
    version_2_with_negative: bool = False,
    n_best_size: int = 20,
    max_answer_length: int = 30,
    null_score_diff_threshold: float = 0.0,
    output_dir: Optional[str] = None,
    prefix: Optional[str] = None,
    log_level: Optional[int] = None,
):
    """
    transformers 기반 모델에서 도출된 logit값들에 대하여 어떤 값을 best로 뽑을지 정의하고, 
    text, probability로 전처리한 후 파일로 저장됨.  
    """
    assert len(predictions) == 2, "`predictions` should be a tuple with two elements (start_logits, end_logits)."
    #predictions : tuple(array[[]], array[[]]) 형식으로 출력됨. (2220, 512) (2220, 512)
    #predictions : transformer 기반 모델에서 나온 마지막 embedding에서 softmax를 통과하기 전 값들임. 
    all_start_logits, all_end_logits = predictions

    #input으로 넣어줬던 eval dataset의 전처리 후 개수와 output인 prediction개수가 같아야 함. 
    assert len(predictions[0]) == len(features), f"Got {len(predictions[0])} predictions and {len(features)} features."

    # Build a map example to its corresponding features.
    """
    원본 데이터의 "guid"가 각 데이터마다 부여되는 고유번호라면,   
    전처리 전 데이터의 index와 원본데이터의 index를 이어주기 위해 example_id_to_index를 생성.
    features_per_example key : guid에 적힌 고유번호, value : 해당 고유번호의 원본 데이터의 index
    """
    example_id_to_index = {k: i for i, k in enumerate(examples["guid"])}
    #딕셔너리(dictionary)와 거의 비슷하지만 key값이 없을 경우 미리 지정해 놓은 초기(default)값을 반환하는 dictionary
    #여기서는 초기 값을 list 타입으로 정함. 
    features_per_example = collections.defaultdict(list)
    """
    example_id에는 tokenized_examples에 대응되는 원본 데이터의 guid 값이 저장되어 있었음.
     ex) feature["example_id"]: [klue-mrc-v1_dev_01842, klue-mrc-v1_dev_03566"...]
     features_per_example key : 원본 데이터 index, value : 해당 원본 데이터의 문서를 전처리 한 데이터가 여러개일 수 있어서, 이를 한번에 list에 저장.
     ex) 1365: [2073, 2074], 1366: [2075, 2076], 1367: [2077, 2078], 1368: [2079, 2080], 1369: [2081, 2082], 1370: [2083, 2084], 1371: [2085, 2086]...
    """
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    # collections.OrderedDict() : dict에 값을 저장할 때마다 그 저장 순서를 보장해 줌. 
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    if version_2_with_negative: #False
        scores_diff_json = collections.OrderedDict()

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        # 원본데이터의 index에 대응되는 전처리 후 데이터의 indices 가져옴. 
        feature_indices = features_per_example[example_index]

        min_null_prediction = None
        prelim_predictions = []

        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            # 512 길이의 예측값을 start, end에 대해서 각각 가져옴. 
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]

            # This is what will allow us to map some the positions in our logits to span of texts in the original context.
            # offset_mapping: 전처리 후 데이터의 offset_mapping 가져오기
            offset_mapping = features[feature_index]["offset_mapping"]

            # Optional `token_is_max_context`, if provided we will remove answers that do not have the maximum context
            # available in the current feature.
            ## 사용하지 않음. 
            token_is_max_context = features[feature_index].get("token_is_max_context", None)

            # Update minimum null prediction.
            # 각각 start classifier와 end classifier에서 출력된 결과값들 중 [CLS] token 자리값만 가져와 더함.  
            feature_null_score = start_logits[0] + end_logits[0]
            if min_null_prediction is None or min_null_prediction["score"] > feature_null_score:
                #min_null_prediction값이 없어서 초기화하거나 이전 값보다 클 경우 업데이트 
                min_null_prediction = {
                    "offsets": (0, 0), #[CLS] token의 offset 
                    "score": feature_null_score,
                    "start_logit": start_logits[0],
                    "end_logit": end_logits[0],
                }

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            #start_indexes, end_indexes: start_logits 속 값에 대하여 오름차순으로 정렬한 후 뒤에서 부터 n_best_size만큼 잘라서 가져옴. 
            #즉, 512개의 토큰 중 가장 큰 값을 가지는 상위 20개의 softmax 값을 가져와 토큰 index를 저장함. 
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            
            # 상위 20개로 뽑은 모든 start, end 쌍 경우의 수에 대하여 
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping) #해당 토큰이 512의 범위를 벗어날 경우 
                        or end_index >= len(offset_mapping) 
                        or offset_mapping[start_index] is None #해당 토큰을 문서에서 참조했을 때 없는 경우  
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    # end index가 start_index보다 작거나 그 차이가 512보다 큰 경우 (문서 전체가 답인 abnormal한 경우)
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    # Don't consider answer that don't have the maximum context available (if such information is
                    # provided).
                    ## 사용하지 않음. 
                    if token_is_max_context is not None and not token_is_max_context.get(str(start_index), False):
                        continue
                    # 총 400개의 start, end 쌍의 경우의 수에 대하여 저장됨. 
                    prelim_predictions.append(
                        {
                            "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]), #원본 문서내에서의 답의 범위 index
                            "score": start_logits[start_index] + end_logits[end_index], #답의 예측 값
                            "start_logit": start_logits[start_index],
                            "end_logit": end_logits[end_index],
                        }
                    )
        if version_2_with_negative: #False
            # Add the minimum null prediction
            prelim_predictions.append(min_null_prediction)
            null_score = min_null_prediction["score"]

        # Only keep the best `n_best_size` predictions.
        # 결국 하나의 문서에 내에서 여러개로 분리되었던 문서 span들의 가능한 score 중, 가장 score가 높은 것을 n_best_size뽑아 저장. 
        predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]

        # Add back the minimum null prediction if it was removed because of its low score.
        if version_2_with_negative and not any(p["offsets"] == (0, 0) for p in predictions):  #False
            predictions.append(min_null_prediction)

        # Use the offsets to gather the answer text in the original context.
        # 문서들의 집합 가져오기. 
        context = example["context"]
        for pred in predictions:
            offsets = pred.pop("offsets")
            # 각 n_best_size개의 predictions에 text 키를 추가하여 문서내 답 str을 저장함. 
            pred["text"] = context[offsets[0] : offsets[1]]

        # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid failure.
        if len(predictions) == 0 or (len(predictions) == 1 and predictions[0]["text"] == ""):
            predictions.insert(0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0})

        # Compute the softmax of all scores (we do it with numpy to stay independent from torch/tf in this file, using
        # the LogSumExp trick).
        # scores : n_best_size개의 score를 한 list에 담음. 
        scores = np.array([pred.pop("score") for pred in predictions])
        # softmax
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        # Include the probabilities in our predictions.
        for prob, pred in zip(probs, predictions):
             # 각 n_best_size개의 predictions에 probability 키를 추가하여 확률값을 저장함. 
            pred["probability"] = prob

        # Pick the best prediction. If the null answer is not possible, this is easy.
        if not version_2_with_negative:
            # 원본데이터에서 guid에 적힌 고유 번호로 key를 저장하고, 가장 높은 점수의 text만을 저장함. 
            all_predictions[example["guid"]] = predictions[0]["text"]
        else:  ## 사용하지 않음. 
            # Otherwise we first need to find the best non-empty prediction.
            i = 0
            while predictions[i]["text"] == "":
                i += 1
            best_non_null_pred = predictions[i]

            # Then we compare to the null prediction using the threshold.
            score_diff = null_score - best_non_null_pred["start_logit"] - best_non_null_pred["end_logit"]
            scores_diff_json[example["guid"]] = float(score_diff)  # To be JSON-serializable.
            if score_diff > null_score_diff_threshold:
                all_predictions[example["guid"]] = ""
            else:
                all_predictions[example["guid"]] = best_non_null_pred["text"]

        # Make `predictions` JSON-serializable by casting np.float back to float.
        # 원본데이터에서 guid에 적힌 고유 번호로 key를 저장하고 value로는 dict 타입으로 모든 predictions내의 offsets, score, start_logit, end_logit, text, probability의 값들을 저장. 
        all_nbest_json[example["guid"]] = [
            {k: (float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v) for k, v in pred.items()}
            for pred in predictions
        ]

    # If we have an output_dir, let's save all those dicts.
    if output_dir is not None:
        assert os.path.isdir(output_dir), f"{output_dir} is not a directory."

        #원본데이터에서 guid에 적힌 고유 번호로 key를 저장하고, 가장 높은 점수의 text만을 저장함. 
        prediction_file = os.path.join( 
            output_dir, "predictions.json" if prefix is None else f"{prefix}_predictions.json"
        )
        # 원본데이터에서 guid에 적힌 고유 번호로 key를 저장하고 value로는 dict 타입으로 모든 predictions내의 offsets, score, start_logit, end_logit, text, probability의 값들을 저장. 
        nbest_file = os.path.join( 
            output_dir, "nbest_predictions.json" if prefix is None else f"{prefix}_nbest_predictions.json"
        )
        if version_2_with_negative:  ## 사용하지 않음. 
            null_odds_file = os.path.join(
                output_dir, "null_odds.json" if prefix is None else f"{prefix}_null_odds.json"
            )

        with open(prediction_file, "w") as writer:
            writer.write(json.dumps(all_predictions, indent=4) + "\n")
        with open(nbest_file, "w") as writer:
            writer.write(json.dumps(all_nbest_json, indent=4) + "\n")
        if version_2_with_negative:
            with open(null_odds_file, "w") as writer:
                writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return all_predictions