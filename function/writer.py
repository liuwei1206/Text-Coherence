# author = liuwei
# date = 2021-09-21

import os
import json
import numpy as np

def subtokens_to_text(sub_tokens):
    words = []
    tmp_word = []
    for token in sub_tokens:
        if len(tmp_word) == 0:
            tmp_word.append(token)
        else:
            if token.startswith("##"):
                tmp_word.append(token[2:])
            else:
                word = "".join(tmp_word)
                words.append(word)
                tmp_word = [token]

    text = " ".join(words)

    return text


def save_preds_for_text_classification(token_ids, attention_mask, tokenizer, true_labels, pred_labels, file):
    """
    save sequence labelling result into files
    Args:
        token_ids:
        attention_mask:
        tokenizer:
        true_labels:
        pred_labels:
        file:
    """
    error_num = 1
    with open(file, 'w', encoding='utf-8') as f:
        for w_ids, attn_mask, t_label, p_label in zip(token_ids, attention_mask, true_labels, pred_labels):
            tokens = tokenizer.convert_ids_to_tokens(w_ids)
            token_num = int(np.sum(attn_mask))
            tokens = tokens[1:token_num-1] # remove [CLS] and [SEP]

            text = subtokens_to_text(tokens)
            if t_label == p_label:
                f.write("%s\t%s\t%s\t%s\n"%("  ", t_label, p_label, text))
            else:
                f.write("%d\t%s\t%s\t%s\t%d\n"%(error_num, t_label, p_label, text))
                error_num += 1

            f.write("\n")
