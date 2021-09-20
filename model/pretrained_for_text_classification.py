# author = liuwei
# date = 2021-09-19

import os
import python

import torch
import torch.nn as nn

from module.bert_modeling import BertModel
from module.roberta_modeling import RobertaModel
from module.albert_modeling import AlbertModel
from module.xlnet_modeling import XLNetModel

from module.bilstm import BiLSTM
from module.transformer import Transformer

class PretrainedForTextClassification(PreTrainedModel):
    """
    Use Pretrained Model for Classification
    Pretrained Models:
        Bert
        Roberta
        Albert
        XLNet
    mode:
        Document-level classification
        Document-paragraph hierarchy classification
        Document-sentence hierarchy classification
    """
    def __init__(self, config, params):
        super(PretrainedForTextClassification, self).__init__(config=config)

        self.model_type = params.model_type
        if params.model_type.upper() == 'BERT':
            self.text_encoder = BertModel(config)
        elif params.model_type.upper() == 'ROBERTA':
            self.text_encoder = RobertaModel(config)
        elif params.model_type.upper() == 'ALBERT':
            self.text_encoder = AlbertModel(config)
        elif params.model_type.upper() == 'XLNET':
            self.text_encoder = XLNetModel(config)

        self.pooling_type = params.pooling_type

        # for document-paragraph hierarchy and document-sentence hierarchy
        if params.doc_type.upper() == "DOCUMENT-PARAGRAPH":
            self.para_transformer = Transformer(
                d_model=params.para_d_model,
                nhead=params.para_nhead,
                num_layers=params.para_num_layers
            )

            if self.para_pooling_type.upper() == "ATTENTION":
                self.para_attn_W1 = nn.Linear(config.hidden_size, config.hidden_size)
                self.para_attn_w2 = nn.Parameter(torch.zeros(config.hidden_size))

        elif params.doc_type.upper() == "DOCUMENT-SENTENCE":
            self.sent_transformer = Transformer(
                d_model=params.sent_d_model,
                nhead=params.sent_nhead,
                num_layers=params.sent_num_layers
            )

        self.dropout = nn.Dropout(config.HP_dropout)
        self.num_labels = params.num_labels
        self.classifier = nn.Linear(config.hidden_size, params.num_labels)

        self.init_weights()

    def forward(
            self,
            doc_token_ids=None,
            doc_segment_ids=None,
            doc_attention_mask=None,
            doc_para_token_ids=None,
            doc_para_segment_ids=None,
            doc_para_attention_mask=None,
            para_attention_mask=None,
            doc_sent_token_ids=None,
            doc_sent_segment_ids=None,
            doc_sent_attention_mask=None,
            sent_attention_mask=None,
            labels=None,
            mode='document',
            flag='Train'
    ):
        """
        Encode the document, and simply treat text coherence as a classification task
        N: batch size
        DL: Document Length
        PL: Paragraph Length
        PN: Paragraph Number in document
        SL: Sentence Length
        SN: Sentence Number in document
        D: hidden_size
        Args:
            # document level
            doc_token_ids: This is a matrix, with shape [N, DL], whole token ids
                            in the document
            doc_segment_ids: [N, DL], 0 for valid tokens
            doc_attention_mask: [N, DL], 1 for valid tokens

            # document-paragraph hierarchy
            doc_para_token_ids: This is a matrix, with shape [N, PN, PL], each row
                                corresponding to the token ids in a paragraph
                                of the doc.
            doc_para_segment_ids: this is a matrix, with shape [N, PN, PL],
                                    each raw corresponding to the segment
                                    ids of the paragraph
            doc_para_attention_mask: a mask matrix, with shape [N, PN],
                                    1 means valid paragraph
            para_attention_mask: a mask matrix, with shape [N, PN, PL],
                                each row corresponding to the valid token
                                in a paragraph of the doc

            # document-sentence hierarchy
            doc_sent_token_ids: a matrix, [N, SN, SL]
            doc_sent_segment_ids: a matrix, [N, SN, SL]
            doc_sent_attention_mask: a vector, [N, SN, SL]
            sent_attention_mask: a matrix, [N, SN, SL]

            labels: [N]
            mode:
                document
                document-paragraph
                document-sent
            flag:
                Train: for training
                Test: for evaluation
        """
        if mode.upper() == "DOCUMENT":
            outputs = self.text_encoder(
                input_ids=doc_token_ids,
                attention_mask=doc_attention_mask,
                token_type_ids=doc_segment_ids
            )

            doc_outputs = outputs[0]

        elif mode.upper() == "DOCUMENT-PARAGRAPH":
            ## 1. for paragraph encoder
            # [N, PM, PL]
            batch_size = doc_para_token_ids.size(0)
            para_num = doc_para_token_ids.size(1)
            para_len = doc_para_token_ids.size(2)
            doc_para_token_ids = doc_para_token_ids.view(batch_size, -1)
            doc_para_segment_ids = doc_para_segment_ids.view(batch_size, -1)
            para_attention_mask = para_attention_mask.view(batch_size, -1)
            outputs = self.text_encoder(
                input_ids=doc_para_token_ids,
                attention_mask=para_attention_mask,
                token_type_ids=doc_para_segment_ids
            )

            sequence_outputs = outputs[0] # [N, PM * PL, D]
            sequence_outputs = sequence_outputs.view(
                batch_size, para_num, para_len, -1
            ) # [N, PM, PL, D]

            ## 2. pooling to get paragraph representation
            # we need to pooling to get the representation of each paragraph
            # convert [N, PM * PL, D] -> [N, PM, D]
            if self.para_pooling_type == "CLS":
                # [CLS] token is always at position 0
                para_outputs = torch.index_select(
                    sequence_outputs, dim=2, index=torch.tensor([0])
                )
                para_outputs = para_outputs.squeeze() # [N, PM, D]

            elif self.para_pooling_type.upper() == "AVERAGE":
                para_attention_mask = para_attention_mask.view(
                    batch_size, para_num, para_len
                )
                para_attention_mask = para_attention_mask.unsqueeze(-1)
                masked_sequence_outputs = sequence_outputs * para_attention_mask.float()
                masked_sequence_outputs = torch.sum(masked_sequence_outputs, dim=2)
                sequence_length = torch.sum(para_attention_mask, dim=2)
                sequence_length = torch.where(
                    sequence_length < 1.0, 1.0, sequence_length
                )
                para_outputs = masked_sequence_outputs / sequence_length # [N, PM, D]
            elif self.para_pooling_type.upper() == "MAX":
                para_attention_mask = para_attention_mask.view(
                    batch_size, para_num, para_len
                )
                para_attention_mask = para_attention_mask.unsqueeze(-1)
                para_attention_mask = para_attention_mask.expand_as(sequence_outputs)
                mask_values = (1 - para_attention_mask.float()) * (-10000.0)
                masked_sequence_outputs = sequence_outputs + mask_values
                masked_sequence_outputs = masked_sequence_outputs.view(
                    batch_size * para_num, para_len, -1) # [N * PM, PL, D]
                masked_sequence_outputs = torch.permute(
                    masked_sequence_outputs, [0, 2, 1]) # [N * PM, D, PL]
                max_pooling = torch.nn.MaxPool1d(kernel_size=para_len)
                masked_sequence_outputs = max_pooling(
                    masked_sequence_outputs) # [N * PM, D]
                para_outputs = masked_sequence_outputs.view(
                    batch_size, para_num, -1) # [N, PM, D]

            ## 3. paragraph transformer
            para_sequence_outputs = self.para_transformer(
                src=para_outputs, mask=doc_para_attention_mask
            )

            if self.doc_pooling_type.upper() == "CLS":
                # in this strategy, simply pick the first one
                doc_outputs = torch.index_select(
                    para_sequence_outputs, dim=1, index=torch.tensor([0])
                )
            elif self.doc_pooling_type.upper() == "AVERAGE":
                para_length = torch.sum(doc_para_attention_mask, dim=1)
                para_length = torch.where(
                    para_length < 1.0, 1.0, para_length
                )
                doc_para_attention_mask = doc_para_attention_mask.unsqueeze(-1)
                masked_para_sequence_outputs = para_sequence_outputs * doc_para_attention_mask
                masked_para_sequence_outputs = torch.sum(
                    masked_para_sequence_outputs, dim=1
                ) # [N, D]
                doc_outputs = masked_para_sequence_outputs / para_length # [N, D]
            elif self.doc_pooling_type.upper() == "ATTENTION":
                alpha = self.para_attn_W1(para_sequence_outputs) # [N, PN, D]
                tan = nn.Tanh()
                alpha = tan(alpha)
                alpha = torch.matmul(alpha, self.para_attn_w2) # [N, PN, D]
                alpha = torch.sum(alpha, dim=2).squeeze() # [N, PN]
                zero_mask = (1 - doc_para_attention_mask.float()) * (-10000.0)
                alpha = alpha + zero_mask
                alpha = torch.nn.Softmax(dim=-1)(alpha)  # [N, PN]
                alpha = alpha.unsqueeze(-1) # [N, PN, 1]
                doc_outputs = torch.sum(
                    para_sequence_outputs * alpha, dim=1
                ) # [N, D]
        elif mode.upper() == "DOCUMENT-SENTENCE":
            pass




