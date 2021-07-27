import os
import argparse
import json
import time
from datetime import datetime
import numpy as np

import torch

from datasets import load_dataset, load_metric, DatasetDict
from transformers import (
    AutoTokenizer,
    EvalPrediction, 
    TrainingArguments, 
    AutoModelForQuestionAnswering,
    default_data_collator,
)

from utils import ( set_seed, postprocess_qa_predictions )
from model.MRCTranier import QuestionAnsweringTrainer

from network import (
    CONFIG_CLASSES,
    TOKENIZER_CLASSES,
    #MODEL_FOR_QUESTION_ANSWERING,
    ElectraForQuestionAnswering,
)

"""
실행 명령어
python main.py
"""

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int, help='Seed')
parser.add_argument('--model', default='klue/bert-base', type=str, help='klue/bert-base, monologg/kobert, monologg/distilkobert, monologg/koelectra-base-v3-discriminator')
parser.add_argument('--task', default=0, type=int, help='MRC')
parser.add_argument('--output_dir', default='checkpoint/', type=str, help='Checkpoint directory/')
parser.add_argument('--result_dir', default='results/', type=str, help='Result directory/')
parser.add_argument('--lr', default=5e-5, type=float, help='Learning rate [1e-5, 2e-5, 3e-5, 5e-5]')
parser.add_argument('--wr', default=0., type=float, help='warm-up ratio [0., 0.1, 0.2, 0.6]')
parser.add_argument('--wd', default=0., type=float, help='weight decay coefficient [0.0, 0.01]')
parser.add_argument('--batch_size', default=8, type=int, help='batch size [8, 16, 32]')
parser.add_argument('--total_epochs', default=3, type=int, help='number of epochs [3, 4, 5, 10]')
parser.add_argument('--train_ratio', default=50, type=int, help='proportion to take for train dataset')
parser.add_argument('--valid_ratio', default=25, type=int, help='proportion to take for validation dataset (this value shoud be lower than 50)')
parser.add_argument('--resume', default=None, type=str, help='resume from checkpoint. ex) 270851bert-base/checkpoint-500/')
parser.add_argument('--tokenizer', default=None, type=str, help='get vocab.txt to generate tokenizer')
parser.add_argument('--customizing', default=False, type=bool, help='customizing...')
parser.add_argument('--freezing', default=False, type=bool, help='Freezing...')
parser.add_argument('--do_lower_case', default=True, type=bool, help='one of tokenizer argument')

p_args = parser.parse_args()
assert p_args.valid_ratio <= 50

start = time.time()

set_seed(p_args.seed)
if not os.path.exists(p_args.result_dir):
    os.makedirs(p_args.result_dir)

KLUE_TASKS = ["mrc"]
KLUE_TASKS_REGRESSION = [False]
task_to_keys = {
    "mrc": (None ,None),
}
sentence1_key, sentence2_key = task_to_keys[KLUE_TASKS[p_args.task]]
max_sequence_length = 512 if KLUE_TASKS[p_args.task] == "mrc" else 128
is_regression = KLUE_TASKS_REGRESSION[p_args.task]
label_column_name = "label"

def compute_metrics(p: EvalPrediction):
    """
    post_process_function의 리턴값인 EvalPrediction을 인풋으로 받아서
    exact, f1, total이 포함된 dict 타입의 result 리턴
    return {
                "exact": 100.0 * sum(exact_raw.values()) / total,
                "f1": 100.0 * sum(f1_raw.values()) / total,
                "total": total
            }
    """
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = metric.compute(predictions=preds, references=p.label_ids)
    if len(result) > 1:
        result["combined_score"] = np.mean(list(result.values())).item()
    return result
    
## Prepare Dataset
data_dir = f"data/{KLUE_TASKS[p_args.task]}"
data_files = {"train": [], "validation": [], "test":[]}
data_files["train"].append(f"{data_dir}/train.json")
data_files["validation"].append(f"{data_dir}/validation.json")
data_files["test"].append(f"{data_dir}/validation.json")
half = int(5841*0.5)
## returned as datasets.Dataset
datasets = load_dataset("json", data_dir=data_dir, data_files=data_files, field='data', split=[f'train[:{p_args.train_ratio}%]', f'validation[:{p_args.valid_ratio}%]', f'test[{half}:]'])
datasets = DatasetDict({"train": datasets[0], "validation": datasets[1], "test": datasets[2] })

# Load the metric
## returned as datasets.Metric 
metric = load_metric('./metric.py', KLUE_TASKS[p_args.task])

## Prepare Pre-trained Model
if not p_args.customizing:
    model = AutoModelForQuestionAnswering.from_pretrained(f"{p_args.model}")
else : 
    config = CONFIG_CLASSES["koelectra-small-v3"].from_pretrained(
        p_args.model,
    )
    tokenizer = TOKENIZER_CLASSES["koelectra-small-v3"].from_pretrained(
        p_args.model,
        do_lower_case=p_args.do_lower_case,
    )
    model = ElectraForQuestionAnswering.from_pretrained(
        p_args.model,
        config=config,
        freeze_electra=p_args.freezing
    )
## Preprocessing the data
# Tokenize all texts and align the labels with them.
def prepare_train_features(examples):
    # Preprocessing is slightly different for training and evaluation
    """
    <Transformers Tokenizer>
    https://paddlenlp-en.readthedocs.io/en/latest/_modules/paddlenlp/transformers/tokenizer_utils.html
    - text
    - text-pair
    - truncation : (only_first)-This will only truncate the first sequence of a pair 
                   (only_second)-This will only truncate the second sequence of a pair 
    - stride : only available for QA usage. the text-pair(context) will be split into multiple spans thus will produce a bigger batch than inputs 

    - return_overflowing_tokens : return List of overflowing tokens in the returned dictionary.
    - return_offsets_mapping : return list of pair preserving the index of start and end char in original input for each token.
    - padding : Pad to a maximum length

    <Example>
    question length : 34
    context length : 1873
    max_length, stride : 512, 128
    한 지문의 토큰이 512보다 많은 경우,
    [0, 512], [512-128, 512*2-128], [512*2-128, 512*3-128].. [512*n-128, pad] 이런식으로 여러개의 tokenized examples를 생성한다.  

    tokenized_examples["input_ids"] : [CLS] + 문제 토큰 + [SEP] + 지문 토큰 + [SEP] 총 길이 512 
    tokenized_examples["token_type_ids"] : 지문 토큰 + [SEP] 부분만 1이고 나머지는 0 
    tokenized_examples["offset_mapping"] : list of pair preserving the index of start and end char in original input for each token.

    [CLS]성공적인 성과를 보인 지역SW서비스사업화 지원사업의 주최자는? 
    ['[CLS]', '성공', '##적인', '성과', '##를', '보인', '지역', '##S', '##W', '##서비스', '##사업', '##화', '지원', '##사업', '##의', '주최', '##자'...
    [(0, 0), (0, 2), (2, 4), (5, 7), (7, 8), (9, 11), (12, 14), (14, 15), (15, 16), (16, 19), (19, 21), (21, 22), (23, 25), (25, 27), (27, 28)
    """

    question_column_name = "question"
    context_column_name = "context"
    answer_column_name = "answers"

    # Padding side determines if we do (question|context) or (context|question)
    ## pad_on_right이 True 이면 question, context 순으로 배치되고 context에 대해서만 truncation이 이루어짐. 
    pad_on_right = tokenizer.padding_side == "right"
    
    ## dict type으로 리턴. 
    ## tokenized_examples dict keys: (['input_ids', 'token_type_ids', 'attention_mask', 'offset_mapping', 'overflow_to_sample_mapping']) 
    tokenized_examples = tokenizer(
        examples[question_column_name if pad_on_right else context_column_name],
        examples[context_column_name if pad_on_right else question_column_name],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_sequence_length, #512
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    """
    sequence_ids:  [CLS], [SEP] 토큰은 None, 문제는 0, 지문은 1
    start_char, end_char: 원본 지문 길이를 기준으로 (토큰화가 되어있다고 가정했을 때)답이 시작하는 index + 답의 길이. 이렇게 해야 offset으로 참조할 때 char 개수만큼 offset이 생성될 가능성을 보장함. 
    offset: Span된 토큰의 index 위치가 원본 지문 길이를 기준으로 어느 위치에 있는지 matching해줄 수 있음. 
    start_positions, end_positions: span된 지문 길이를 기준으로 
    지문 span 생성하다보면 내부에 답이 없는 경우가 있기 때문에 그런 경우는 정답 토큰을 cls로 지정함. 
    answer가 multi label인데 첫번째 것만 answer로 사용함. 
    """

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS tokens
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        ## sequence_ids : Return a list mapping the tokens to the id of their original sentences
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        ## sample_index : 현재 span된 데이터가 몇번째 원본 데이터를 참조하는 지 저장. 
        sample_index = sample_mapping[i]
        answers = examples[answer_column_name][sample_index]
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])
            
            # Start token index of the current span in the text.
            ## (1 if pad_on_right else 0): right에 context이면 1 
            ## token_start_index를 지문의 시작점으로 지정하는 부분
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # End token index of the current span in the text.
            ## (1 if pad_on_right else 0): right에 context이면 1 
            ## token_end_index 지문의 끝점으로 지정하는 부분
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                ## offsets[token_start_index][0]가 start_char가 될때까지 
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                ## offsets[token_end_index][1]가 end_char가 될때까지 
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples

def prepare_validation_features(examples):
    """
    training data는 예측값(start_positions, end_positions)를 만들어 학습해야 하지만
    validation data는 바로 예측한 결과를 텍스트로 바꿔 평가하면 되어서 (start_positions, end_positions)를 만들어줄 필요가 없음.

    BERT 인풋에 넣을 tokenized_examples["input_ids"]를 만들기 위한 tokenizer만 선언해줘도 됨. 
    """
    # Preprocessing is slightly different for training and evaluation
    question_column_name = "question"
    context_column_name = "context"
    answer_column_name = "answers"

    # Padding side determines if we do (question|context) or (context|question)
    ## pad_on_right이 True 이면 question, context 순으로 배치되고 context에 대해서만 truncation이 이루어짐. 
    pad_on_right = tokenizer.padding_side == "right"

    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.

    ## train_features에서 한 것과 동일 
    tokenized_examples = tokenizer(
        examples[question_column_name if pad_on_right else context_column_name],
        examples[context_column_name if pad_on_right else question_column_name],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_sequence_length,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
    # corresponding example_id and we will store the offset mappings.
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        ## example_id에는 tokenized_examples에 대응되는 원본 데이터의 guid 값을 저장함. 
        tokenized_examples["example_id"].append(examples["guid"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        ## 토큰화된 지문에서 '지문'에 해당하는 자리이면 offset을 저장하고 그렇지 않으면 None을 저장함. 
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples

# Post-processing:
def post_processing_function(examples, features, predictions, stage="eval"):
    """
    MRCTrainer에서 다음과 같이 사용됨.
    eval_preds = self.post_process_function(eval_examples, eval_dataset, output.predictions)
    
    formatted_predictions: 원본 데이터 하나마다 예측한 텍스트를 쌍으로 저장
    references: 원본 데이터 하나마다 example["answers"]항목을 쌍으로 저장 
        ex) formatted_predictions {'guid': 'klue-mrc-v1_dev_03199', 'prediction_text': '벨기에'} 
            references {'guid': 'klue-mrc-v1_dev_03199', 'answers': {'answer_start': [158], 'text': ['벨기에']}}
    <EvalPrediction>
    Evaluation output (always contains labels), to be used to compute metrics.
    """
    answer_column_name = "answers"

    # Post-processing: we match the start logits and end logits to answers in the original context.
    predictions = postprocess_qa_predictions(
        examples=examples,
        features=features,
        predictions=predictions,
        version_2_with_negative=False,
        n_best_size=20,
        max_answer_length=30,
        null_score_diff_threshold=0.0,
        output_dir="data/mrc/",
        prefix=stage,
    )
    # Format the result to the format the metric expects.
    formatted_predictions = [{"guid": k, "prediction_text": v} for k, v in predictions.items()]

    references = [{"guid": ex["guid"], "answers": ex[answer_column_name]} for ex in examples]
    return EvalPrediction(predictions=formatted_predictions, label_ids=references)

if p_args.tokenizer is None:
    tokenizer = AutoTokenizer.from_pretrained(f"{p_args.model}")
else:
    tokenizer = AutoTokenizer.from_pretrained(f"{p_args.model}")

"""
<datasets.Dataset.map>
-function
-batched: Provide batch of exampes fo function 
-remove_columns: columns will be removed before updating the examples with the output of function

<transformers.data.data_collator.default_data_collators>
https://huggingface.co/transformers/main_classes/data_collator.html
"""
column_names = datasets["train"].column_names
train_examples = datasets["train"]
train_dataset = train_examples.map(prepare_train_features, batched=True, remove_columns=column_names)
print(f"# of Original Train Dataset : 17554") #17554
print(f"# of Splitted Train Dataset : {len(train_examples)}") #8777

column_names = datasets["validation"].column_names
validation_examples = datasets["validation"]
test_examples = datasets["test"]
validation_dataset = validation_examples.map(prepare_validation_features, batched=True, remove_columns=column_names)
test_dataset = test_examples.map(prepare_validation_features, batched=True, remove_columns=column_names)
data_collator = (default_data_collator)
print(f"# of Valid dataset : {len(validation_examples)}, Test dataset : {len(test_examples)}") #2920, 2921

now = datetime.now()
now_str = f'{now.day:02}{now.hour:02}{now.minute:02}'

if p_args.resume is None:
    output_dir_name = p_args.output_dir+now_str+p_args.model.split('/')[1]+'/'
    ckpt_dir_name = None
else :
    output_dir_name =  p_args.output_dir
    ckpt_dir_name = p_args.output_dir+p_args.resume

args = TrainingArguments(
        output_dir=output_dir_name,
        evaluation_strategy='epoch',
        learning_rate=p_args.lr,
        per_device_train_batch_size=p_args.batch_size,
        per_device_eval_batch_size=p_args.batch_size,
        num_train_epochs=p_args.total_epochs,
        weight_decay=p_args.wd,
        warmup_ratio=p_args.wr,
        seed=p_args.seed,
        save_total_limit=1,
        logging_strategy="no",
    )

"""
validation_dataset: 전처리 후 데이터
validation_examples: 전처리 전 데이터 
"""
trainer = QuestionAnsweringTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    eval_examples=validation_examples,
    tokenizer=tokenizer,
    data_collator=data_collator,
    post_process_function=post_processing_function,
    compute_metrics=compute_metrics,
)

trainer.train(resume_from_checkpoint=ckpt_dir_name)
trainer.evaluate(
    eval_dataset=test_dataset,
    eval_examples=test_examples,
    metric_key_prefix='test'
)

log_history = trainer.state.log_history

elapsed_time = (time.time() - start) / 60 # Min.

model_name = p_args.model.split('/')[1]
path = f'{p_args.result_dir}model_{model_name}_{now_str}.json'
mode = 'a' if os.path.isfile(path) else 'w'

hyper_param = {"learning_rate": p_args.lr, "warmup_ratio": p_args.wr, "weight_decay": p_args.wd, "batch_size": p_args.batch_size, "epochs":p_args.total_epochs}
with open(path, mode) as f:
    result = {
        'seed': p_args.seed,
        'hyper_param': hyper_param,
        f'{KLUE_TASKS[p_args.task]}': log_history,
        'time': elapsed_time,
    }
    json.dump(result, f, indent=2)

print("\nFinished...")
