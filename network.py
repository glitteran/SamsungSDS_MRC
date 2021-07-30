#-*- coding: utf-8 -*-
from transformers.models.electra.modeling_electra import (
    ElectraModel, ElectraPreTrainedModel
)
from transformers.models.bert.modeling_bert import (
    BertModel, BertPreTrainedModel
)
from transformers.models.albert.modeling_albert import (
    AlbertModel, AlbertPreTrainedModel
)

from transformers import (
    ElectraConfig, ElectraTokenizer, ElectraForQuestionAnswering,
    AlbertConfig, AlbertTokenizer, AlbertForQuestionAnswering,
    BertConfig, BertTokenizer, # BertForQuestionAnswering,
)

from transformers.modeling_outputs import (
    QuestionAnsweringModelOutput,
)

from torch import nn

from torch.nn import CrossEntropyLoss

"""
참고 자료
1. https://github.com/monologg/KoELECTRA/tree/master/finetune
2. https://huggingface.co/transformers/_modules/transformers/models/electra/modeling_electra.html#ElectraForQuestionAnswering
3. https://github.com/huggingface/transformers/tree/master/src/transformers
"""

CONFIG_CLASSES = {"koelectra-small-v3": ElectraConfig,'albert':AlbertConfig,'klue/bert-base':BertConfig}
TOKENIZER_CLASSES = {"koelectra-small-v3": ElectraTokenizer,'albert':AlbertTokenizer,'klue/bert-base':BertTokenizer}
MODEL_FOR_QUESTION_ANSWERING = {"koelectra-small-v3":ElectraForQuestionAnswering, 'albert':AlbertForQuestionAnswering,'klue/bert-base':BertForQuestionAnswering}

class ElectraForQuestionAnswering(ElectraPreTrainedModel):
    #https://github.com/huggingface/transformers/blob/master/src/transformers/models/electra/configuration_electra.py
    config_class = ElectraConfig
    base_model_prefix = "electra"

    def __init__(self, config, freeze_electra: bool = False):
        super().__init__(config)
        """
        config:
        모델을 pre-training하면서 고정했던 값들
        hidden_size, num_attention_heads, hidden_dropout_prob.. 등등 
        """
        self.num_labels = config.num_labels #2
        self.electra = ElectraModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels) #(768, 2)

        """
        self.init_weights()을 사용하는 이유: 
        1. class내 매개변수(weight)에 대하여 pseudo-random initialization 을 진행 
        2. .from_pretrained("koelectra-small-v3")를 통해 요청한 모델의 기학습된 매개변수(weight)을 불러와 오버라이팅함. 
        결과적으로 pre-train 되지 않은 말단 layer에 대해서는 random initialization하는 것을 보장함. 
        """
        self.init_weights()

        if freeze_electra:
            for param in self.base_model.parameters():
                param.requires_grad = False

    # @add_start_docstrings_to_model_forward(ELECTRA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # @add_code_sample_docstrings(
    #     tokenizer_class=_TOKENIZER_FOR_DOC,
    #     checkpoint=_CHECKPOINT_FOR_DOC,
    #     output_type=QuestionAnsweringModelOutput,
    #     config_class=_CONFIG_FOR_DOC,
    # )

    # class ElectraModel(ElectraPreTrainedModel)
    # https://huggingface.co/transformers/_modules/transformers/models/electra/modeling_electra.html#ElectraModel
    # forward 인자 설명 : https://huggingface.co/transformers/model_doc/electra.html
    def forward(
        self,
        input_ids=None, # (batch_size, sequence_length) Indices of input sequence tokens in the vocabulary.
        attention_mask=None, # (batch_size, sequence_length) Mask to avoid performing attention on padding token indices. 
        token_type_ids=None, # (batch_size, sequence_length) Segment token indices to indicate first and second portions of the inputs. 0 is sentence A, 1 is sentence B
        position_ids=None, # None (batch_size, sequence_length) Indices of positions of each input sequence tokens in the position embeddings.
        head_mask=None,  # None (num_layers, num_heads) Mask to nullify selected heads of the self-attention modules.
        inputs_embeds=None, # None (batch_size, sequence_length, hidden_size) embedded representation instead of input_ids 
        start_positions=None, # Labels for position (index) of the start of the labelled span for computing the token classification loss.
        end_positions=None, # Labels for position (index) of the end of the labelled span for computing the token classification loss.
        output_attentions=None, # None, Whether or not to return the attentions tensors of all attention layers
        output_hidden_states=None, # None, Whether or not to return the hidden states of all layers. 
        return_dict=None, # None, Whether or not to return a ModelOutput instead of a plain tuple.
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        discriminator_hidden_states = self.electra( # torch.Size([8, 512, 768]) = (batch, max_length, hidden dim size)
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        # sequence_output = self.attention_layer(discriminator_hidden_states)
        #print("discriminator_hidden_states",discriminator_hidden_states[0].size())
        sequence_output = discriminator_hidden_states[0]

        logits = self.qa_outputs(sequence_output) #torch.Size([8, 512, 2])   
        start_logits, end_logits = logits.split(1, dim=-1) #torch.Size([8, 512, 1]) , torch.Size([8, 512, 1]) 
        start_logits = start_logits.squeeze(-1).contiguous() #torch.Size([8, 512]) 
        end_logits = end_logits.squeeze(-1).contiguous()  #torch.Size([8, 512]) 

        total_loss = None
        #start, end 둘 다 정답 값이 존재하는 경우 
        if start_positions is not None and end_positions is not None: 
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            # 각 start_position과 end_position들의 값들 중에 0 미만인 경우 0으로, max length를 넘는 값은 max length으로 변환함.
            ignored_index = start_logits.size(1) #max length를 의미함. 512

            #before clamp : tensor([  0, 461, 149, 214, 373,   0, 309, 106], device='cuda:0')
            #after clasmp:  tensor([  0, 461, 149, 214, 373,   0, 309, 106], device='cuda:0')
            start_positions = start_positions.clamp(0, ignored_index) 
            end_positions = end_positions.clamp(0, ignored_index)

            # 지정 ignored_index에 대해서는 loss값을 구할 때 고려되지 않음. 
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict: ##사용하지 않음
            output = (
                start_logits,
                end_logits,
            ) + discriminator_hidden_states[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        #loss, start_logits, end_logits, hidden_states(None), attentions(None)이 차례로 저장됨. 
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )

class BertForQuestionAnswering(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.gru = nn.GRU(config.hidden_size,int(config.hidden_size/2))
        self.qa_outputs = nn.Linear(int(config.hidden_size/2), config.num_labels)

        self.init_weights()

    # @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # @add_code_sample_docstrings(
    #     tokenizer_class=_TOKENIZER_FOR_DOC,
    #     checkpoint=_CHECKPOINT_FOR_DOC,
    #     output_type=QuestionAnsweringModelOutput,
    #     config_class=_CONFIG_FOR_DOC,
    # )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]



        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )