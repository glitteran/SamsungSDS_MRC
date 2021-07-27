#-*- coding: utf-8 -*-
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
    MODEL_FOR_QUESTION_ANSWERING,
)

"""
실행 명령어
python main.py
"""

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int, help='Seed')
parser.add_argument('--model', default='klue/bert-base', type=str,
                    help='klue/bert-base, monologg/kobert, monologg/distilkobert, monologg/koelectra-base-v3-discriminator ,snunlp/KR-BERT-char16424, ainize/klue-bert-base-mrc')
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
datasets = load_dataset("json", data_dir=data_dir, data_files=data_files, field='data', split=[f'train[:{p_args.train_ratio}%]', f'validation[:{p_args.valid_ratio}%]', f'test[{half}:]'])
datasets = DatasetDict({"train": datasets[0], "validation": datasets[1], "test": datasets[2] })

# Load the metric
metric = load_metric('./metric.py', KLUE_TASKS[p_args.task])

## Prepare Pre-trained Model
if not p_args.customizing:
    model = AutoModelForQuestionAnswering.from_pretrained(f"{p_args.model}")
else : 
    config = CONFIG_CLASSES[p_args.model].from_pretrained(
        p_args.model,
    )
    tokenizer = TOKENIZER_CLASSES[p_args.model].from_pretrained(
        p_args.model,
        do_lower_case=p_args.do_lower_case,
    )
    model = MODEL_FOR_QUESTION_ANSWERING[p_args.model].from_pretrained(
        p_args.model,
        config=config,
    )
## Preprocessing the data
# Tokenize all texts and align the labels with them.
def prepare_train_features(examples):
    # Preprocessing is slightly different for training and evaluation
    question_column_name = "question"
    context_column_name = "context"
    answer_column_name = "answers"

    # Padding side determines if we do (question|context) or (context|question)
    pad_on_right = tokenizer.padding_side == "right"
    
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

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS tokens
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
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
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # End token index of the current span in the text.
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
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples

def prepare_validation_features(examples):
    # Preprocessing is slightly different for training and evaluation
    question_column_name = "question"
    context_column_name = "context"
    answer_column_name = "answers"

    # Padding side determines if we do (question|context) or (context|question)
    pad_on_right = tokenizer.padding_side == "right"

    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
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
        tokenized_examples["example_id"].append(examples["guid"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples

# Post-processing:
def post_processing_function(examples, features, predictions, stage="eval"):
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
