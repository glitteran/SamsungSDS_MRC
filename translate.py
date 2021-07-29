#-*- coding: utf-8 -*-

from transformers import MBartForConditionalGeneration, MBart50Tokenizer, MBart50TokenizerFast
from copy import deepcopy
from transformers import (
    AutoTokenizer,
    EvalPrediction,
    TrainingArguments,
    AutoModelForQuestionAnswering,
    default_data_collator,
)
from datasets import load_dataset, load_metric, DatasetDict
from tqdm import tqdm

# def load_dataset(path)
#     copied = Dataset.load_from_disk('data/copied.json/')

def translate(train_examples, src_lang ="ko_KR", target_lang=['en_XX',],ouput_name='translated') :
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    origin_examples = deepcopy(train_examples)
    for target in target_lang :
        translate_examples=deepcopy(origin_examples)
        for trans_data in tqdm(translate_examples) :
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

    train_examples.save_to_disk(f"{ouput_name}-{'-'.join(target_lang)}")

    return train_examples