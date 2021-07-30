#-*- coding: utf-8 -*-
from transformers import MBartForConditionalGeneration, MBart50Tokenizer, MBart50TokenizerFast
from copy import deepcopy
from datetime import datetime
from transformers import (
    AutoTokenizer,
    EvalPrediction,
    TrainingArguments,
    AutoModelForQuestionAnswering,
    default_data_collator,
)
from datasets import load_dataset, load_metric, DatasetDict
from tqdm import tqdm
import math

# def load_dataset(path)
#     copied = Dataset.load_from_disk('data/copied.json/')

def translate(origin_examples, train_examples=None, src_lang ="ko_KR", target_lang=['en_XX',],output_name='translated') :
    if not train_examples:
        train_examples = deepcopy(origin_examples)
    
    check_i = math.floor(len(train_examples)/len(origin_examples))-1
    check_j = math.floor(len(train_examples)%len(origin_examples))
    checkpoint = 500

    print(f'start translate: {target_lang[check_i:]}, checkpoint {check_j}')

    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

    for target in target_lang[check_i:] :
        translate_examples = deepcopy(origin_examples)
        for trans_data in tqdm(origin_examples.select(range(check_j, len(origin_examples)))) :
            # kor -> target lang
            tokenizer.src_lang = src_lang
            encoded_ko = tokenizer(trans_data['question'], return_tensors="pt")
            generated_tokens = model.generate(**encoded_ko, forced_bos_token_id=tokenizer.lang_code_to_id[target])
            tar_sent = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            # target -> korean
            tokenizer.src_lang = target
            encoded_tar = tokenizer(tar_sent, return_tensors="pt")
            generated_tokens = model.generate(**encoded_tar, forced_bos_token_id=tokenizer.lang_code_to_id[src_lang])
            trans_data['question'] = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            train_examples = train_examples.add_item(trans_data)

            if (len(train_examples)/len(origin_examples)) % checkpoint == 0:
                print(f'save checkpoint: {len(train_examples)}')
                train_examples.save_to_disk(f"checkpoint/{output_name}-{'-'.join(target_lang)}")

        check_j = 0

    now = datetime.now()
    now_str = f'{now.day:02}{now.hour:02}{now.minute:02}'
    train_examples.save_to_disk(f"checkpoint/{output_name}-{'-'.join(target_lang)}-{now_str}")
    print('end translate')

    return train_examples