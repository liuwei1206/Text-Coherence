# author = liuwei
# date = 2021-09-16

import os
import json
import random
import time
import torch
import numpy as np
from torch.utils.data import Dataset
random.seed(123456)

class GCDCDataset(Dataset):
    """
    The dataset instance for GCDC corpus.
    Note, this dataset should support for sentence-document hierarchy, paragraph-document
    hierarchy and document-level modeling.
    """
    def __init__(self, file, params):
        """
        Args:
            file:
            params:
        """
        self.tokenizer = params['tokenizer']  # for BPE or WordPiece
        # for document-level modeling
        self.max_doc_length = params['max_doc_length']
        # for document-paragraph hierarchy modeling
        self.max_para_num = params['max_para_num'] # the max paragraph number
        self.max_para_length = params['max_para_length'] # max length of each paragraph
        # for document-sentence hierarchy modeling
        self.max_sent_num = params['max_sent_num'] # the max sentence number
        self.max_sent_length = params['max_sent_length'] # the max length of each sentence

        self.file = file
        file_items = file.split("/")  # to get data_dir
        data_dir = "/".join(file_items[:-1])
        saved_file_name = "saved-doc_len_{}-max_doc_para_{}_{}-max_doc_sent_{}_{}".format(
            self.max_doc_length, self.max_para_num, self.max_para_length,
            self.max_sent_num, self.max_sent_length
        ) + file_items[-1].split('.')[0] + ".npz"
        saved_np_file = os.path.join(data_dir, saved_file_name)
        self.saved_np_file = saved_np_file

        # read all data and convert then into np format, which can speed the read process
        self.init_np_dataset()

    def init_np_dataset(self):
        """
        obtain the numpy format of the total dataset
        """
        print_flag = True  # print some instance to check if correct
        if os.path.exists(self.saved_np_file):
            with np.load(self.saved_np_file) as dataset:
                self.doc_token_ids = dataset['doc_token_ids']
                self.doc_segment_ids = dataset['doc_segment_ids']
                self.doc_attention_mask = dataset['doc_attention_mask']
                self.doc_para_token_ids = dataset['doc_para_token_ids']
                self.doc_para_segment_ids = dataset['doc_para_segment_ids']
                self.doc_para_attention_mask = dataset['doc_para_attention_mask']
                self.para_attention_mask = dataset['para_attention_mask']
                self.doc_sent_token_ids = dataset['doc_sent_token_ids']
                self.doc_sent_segment_ids = dataset['doc_sent_segment_ids']
                self.doc_sent_attention_mask = dataset['doc_sent_attention_mask']
                self.sent_attention_mask = dataset['sent_attention_mask']

                if print_flag:
                    print("Document instance: ")
                    print("     tokens: ", self.tokenizer.convert_ids_to_tokens(self.doc_token_ids[0][:20]))
                    print("     token_ids: ", self.doc_token_ids[0][:20])
                    print("     segment_ids: ", self.doc_segment_ids[0][:20])
                    print("     attention_mask: ", self.doc_attention_mask[0][:20])

                    print_flag = False
        else:
            # for document-level modeling
            all_doc_token_ids = []
            all_doc_segment_ids = []
            all_doc_attention_mask = []

            # for document-paragraph hierarchy modeling
            all_doc_para_token_ids = []
            all_doc_para_segment_ids = []
            all_doc_para_attention_mask = []
            all_para_attention_mask = []

            # for document-sentence hierarchy modeling
            all_doc_sent_token_ids = []
            all_doc_sent_segment_ids = []
            all_doc_sent_attention_mask = []
            all_sent_attention_mask = []

            with open(self.file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if line:
                        sample = json.loads(line)
                        text = sample['text']
                        label = sample['labelA']

                        # for document-level modeling, document-paragraph, document-sent
                        document_para_text = [" ".join(tt) for tt in text]
                        document_text = " ".join(document_para_text)
                        document_sent_text = []
                        for para in text:
                            for sent in para:
                                document_sent_text.append(sent)

                        ## tokenize
                        # tokenize for document
                        document_text_tokens = self.tokenizer.tokenize(document_text)
                        if len(document_text_tokens) > self.max_doc_length - 2:
                            document_text_tokens = document_text_tokens[:self.max_doc_length-2]
                        document_text_tokens.insert(0, '[CLS]')
                        document_text_tokens.append('[SEP]')
                        document_token_ids = self.tokenizer.convert_tokens_to_ids(document_text)

                        # tokenize for paragraphs
                        document_para_tokens = []
                        document_para_token_ids = []
                        tmp_para_num = 0
                        for para in document_para_text:
                            para_tokens = self.tokenizer.tokenize(para)
                            if len(para_tokens) > self.max_para_length - 2:
                                para_tokens = para_tokens[:self.max_para_length-2]
                            para_tokens.insert(0, '[CLS]')
                            para_tokens.append('[SEP]')
                            document_para_tokens.append(para_tokens)
                            document_para_token_ids.append(self.tokenizer.convert_tokens_to_ids(para_tokens))
                            tmp_para_num += 1
                            if tmp_para_num >= self.max_para_num:
                                break

                        # tokenizer for sentence
                        document_sent_tokens = []
                        document_sent_token_ids = []
                        tmp_sent_num = 0
                        for sent in document_sent_text:
                            sent_tokens = self.tokenizer.tokenize(sent)
                            if len(sent_tokens) > self.max_sent_length - 2:
                                sent_tokens = sent_tokens[:self.max_sent_length-2]
                            sent_tokens.insert(0, '[CLS]')
                            sent_tokens.append('[SEP]')
                            sent_token_ids = self.tokenizer.convert_tokens_to_ids(sent_tokens)
                            document_sent_tokens.append(sent_tokens)
                            document_sent_token_ids.append(sent_token_ids)
                            tmp_sent_num += 1
                            if tmp_sent_num >= self.max_sent_num:
                                break

                        ## to numpy matrix
                        # document
                        document_token_ids_np = np.zeros(self.max_doc_length)
                        document_segment_ids_np = np.ones(self.max_doc_length)
                        document_attention_mask_np = np.zeros(self.max_doc_length)

                        document_token_ids_np[:len(document_token_ids)] = document_token_ids
                        document_segment_ids_np[:len(document_token_ids)] = 0
                        document_attention_mask_np[:len(document_token_ids)] = 1

                        # document-paragraph
                        document_para_token_ids_np = np.zeros(self.max_para_num, self.max_para_length)
                        document_para_segment_ids_np = np.ones(self.max_para_num, self.max_para_length)
                        document_para_attention_mask_np = np.zeros(self.max_para_num)
                        para_attention_mask_np = np.zeros(self.max_para_num, self.max_para_length)

                        tmp_para_num = len(document_para_token_ids)
                        for idx in range(tmp_para_num):
                            document_para_attention_mask_np[idx] = 1
                            document_para_token_ids_np[idx][:len(document_para_token_ids[idx])] = document_para_token_ids[idx]
                            document_para_segment_ids_np[idx][:len(document_para_token_ids[idx])] = 0
                            para_attention_mask_np[idx][:len(document_para_token_ids[idx])] = 1

                        # document-sentence
                        document_sent_token_ids_np = np.zeros(self.max_sent_num, self.max_sent_length)
                        document_sent_segment_ids_np = np.ones(self.max_sent_num, self.max_sent_length)
                        document_sent_attention_mask_np = np.zeros(self.max_sent_num)
                        sent_attention_mask_np = np.zeros(self.max_sent_num, self.max_sent_length)

                        tmp_sent_num = len(document_sent_token_ids)
                        for idx in range(tmp_sent_num):
                            document_sent_attention_mask_np[idx] = 1
                            document_sent_token_ids_np[idx][:len(document_sent_token_ids[idx])] = document_sent_token_ids[idx]
                            document_sent_segment_ids_np[idx][:len(document_sent_token_ids[idx])] = 0
                            sent_attention_mask_np[idx][:len(document_sent_token_ids[idx])] = 1

                        # for label
                        label_id = int(label)

                        if print_flag:
                            print("Document instance: ")
                            print("     tokens: ", document_text_tokens[:20])
                            print("     token_ids: ", document_token_ids[:20])
                            print("     segment_ids: ", document_segment_ids_np[:20])
                            print("     attention_mask: ", document_attention_mask_np[:20])

                            print_flag = False

                        all_doc_token_ids.append(document_token_ids_np)
                        all_doc_segment_ids.append(document_segment_ids_np)
                        all_doc_attention_mask.append(document_attention_mask_np)

                        all_doc_para_token_ids.append(document_para_token_ids_np)
                        all_doc_para_segment_ids.append(document_para_segment_ids_np)
                        all_doc_para_attention_mask.append(document_para_attention_mask_np)
                        all_para_attention_mask.append(para_attention_mask_np)

                        all_doc_sent_token_ids.append(document_sent_token_ids_np)
                        all_doc_sent_segment_ids.append(document_sent_segment_ids_np)
                        all_doc_sent_attention_mask.append(document_sent_attention_mask_np)
                        all_sent_attention_mask.append(sent_attention_mask_np)

            assert len(all_doc_token_ids) == len(all_doc_segment_ids), (len(all_doc_token_ids), len(all_doc_segment_ids))
            assert len(all_doc_token_ids) == len(all_doc_attention_mask), (len(all_doc_token_ids), len(all_doc_attention_mask))
            assert len(all_doc_token_ids) == len(all_doc_para_token_ids), (len(all_doc_token_ids), len(all_doc_para_token_ids))
            assert len(all_doc_token_ids) == len(all_doc_para_segment_ids), (len(all_doc_token_ids), len(all_doc_para_segment_ids))
            assert len(all_doc_token_ids) == len(all_doc_para_attention_mask), (len(all_doc_token_ids), len(all_doc_para_attention_mask))
            assert len(all_doc_token_ids) == len(all_para_attention_mask), (len(all_doc_token_ids), len(all_para_attention_mask))
            assert len(all_doc_token_ids) == len(all_doc_sent_token_ids), (len(all_doc_token_ids), len(all_doc_sent_token_ids))
            assert len(all_doc_token_ids) == len(all_doc_sent_segment_ids), (len(all_doc_token_ids), len(all_doc_segment_ids))
            assert len(all_doc_token_ids) == len(all_doc_sent_attention_mask), (len(all_doc_token_ids), len(all_doc_sent_attention_mask))
            assert len(all_doc_token_ids) == len(all_sent_attention_mask), (len(all_doc_token_ids), len(all_sent_attention_mask))

            all_doc_token_ids = np.array(all_doc_token_ids)
            all_doc_segment_ids = np.array(all_doc_segment_ids)
            all_doc_attention_mask = np.array(all_doc_attention_mask)
            all_doc_para_token_ids = np.array(all_doc_para_token_ids)
            all_doc_para_segment_ids = np.array(all_doc_para_segment_ids)
            all_doc_para_attention_mask = np.array(all_doc_para_attention_mask)
            all_para_attention_mask = np.array(all_para_attention_mask)
            all_doc_sent_token_ids = np.array(all_doc_sent_token_ids)
            all_doc_sent_segment_ids = np.array(all_doc_sent_segment_ids)
            all_doc_sent_attention_mask = np.array(all_doc_sent_attention_mask)
            all_sent_attention_mask = np.array(all_sent_attention_mask)

            np.savez(
                self.saved_np_file,
                doc_token_ids=all_doc_token_ids,
                doc_segment_ids=all_doc_segment_ids,
                doc_attention_mask=all_doc_attention_mask,
                doc_para_token_ids=all_doc_para_token_ids,
                doc_para_segment_ids=all_doc_para_segment_ids,
                doc_para_attention_mask=all_doc_para_attention_mask,
                para_attention_mask=all_para_attention_mask,
                doc_sent_token_ids=all_doc_sent_token_ids,
                doc_sent_segment_ids=all_doc_sent_segment_ids,
                doc_sent_attention_mask=all_doc_sent_attention_mask,
                sent_attention_mask=all_sent_attention_mask
            )

            # init dataset
            self.doc_token_ids = all_doc_token_ids
            self.doc_segment_ids = all_doc_segment_ids
            self.doc_attention_mask = all_doc_attention_mask
            self.doc_para_token_ids = all_doc_para_token_ids
            self.doc_para_segment_ids = all_doc_para_segment_ids
            self.doc_para_attention_mask = all_doc_para_attention_mask
            self.para_attention_mask = all_para_attention_mask
            self.doc_sent_token_ids = all_doc_sent_token_ids
            self.doc_sent_segment_ids = all_doc_sent_segment_ids
            self.doc_sent_attention_mask = all_doc_sent_attention_mask
            self.sent_attention_mask = all_sent_attention_mask

        self.total_size = len(self.doc_token_ids)

    def __len__(self):
        return self.total_size

    def __getitem__(self, index):
        """
        return index instance
        Args:
            index: the index point to the index_th sample in all dataset
        """
        return (
            torch.tensor(self.doc_token_ids[index]),
            torch.tensor(self.doc_segment_ids[index]),
            torch.tensor(self.doc_attention_mask[index]),
            torch.tensor(self.doc_para_token_ids[index]),
            torch.tensor(self.doc_para_segment_ids[index]),
            torch.tensor(self.doc_para_attention_mask[index]),
            torch.tensor(self.para_attention_mask[index]),
            torch.tensor(self.doc_sent_token_ids[index]),
            torch.tensor(self.doc_sent_segment_ids[index]),
            torch.tensor(self.doc_sent_attention_mask[index]),
            torch.tensor(self.sent_attention_mask[index])
        )

















