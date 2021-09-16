# author = liuwei
# date = 2021-09-16

import os
import json
import csv

def doc_to_sents_paras(doc):
    """split doc into paras and sents, shows a hierarchy structure"""
    raw_paragraphs = doc.splitlines()

    paragraphs = []
    for para in raw_paragraphs:
        para = para.strip()
        if para:
            sents = para.split(". ")
            sents = [sent + '.' for sent in sents if sent.strip()]
            sents_2 = []
            for sent in sents:
                if '?' in sent:
                    ss = sent.split("? ")
                    ss = [s for s in ss if s.strip()]
                    num = len(ss)
                    for idx in range(num - 1):
                        sents_2.append(ss[idx] + '?')
                    sents_2.append(ss[-1])
                else:
                    sents_2.append(sent)

            sents_3 = []
            for sent in sents_2:
                if '!' in sent:
                    ss = sent.split("! ")
                    ss = [s for s in ss if s.strip()]
                    num = len(ss)
                    for idx in range(num - 1):
                        sents_3.append(ss[idx] + '!')
                    sents_3.append(ss[-1])
                else:
                    sents_3.append(sent)

            paragraphs.append(sents_3)

    return paragraphs


def read_gcdt_csv_to_json(file, json_file):
    """
    'text_id', 'subject', 'text', 'ratingA1', 'ratingA2', 'ratingA3', 'labelA', 'ratingM1', 'ratingM2', 'ratingM3', 'ratingM4', 'ratingM5', 'labelM'
    0			1 			2 		3 			4 			5 			6 			7			8			9			10			11 			12
    """
    all_texts = []
    heads = []
    line_num = 0
    csv_reader = csv.reader(open(file))
    for line in csv_reader:
        if line_num == 0:
            line_num += 1
            continue

        sample = {}
        sample['text'] = doc_to_sents_paras(line[2])
        sample['ratingA1'] = line[3]
        sample['ratingA2'] = line[4]
        sample['ratingA3'] = line[5]
        sample['labelA'] = line[6]
        sample['ratingM1'] = line[7]
        sample['ratingM2'] = line[8]
        sample['ratingM3'] = line[9]
        sample['ratingM4'] = line[10]
        sample['ratingM5'] = line[11]
        sample['ratingM'] = line[12]

        all_texts.append(json.dumps(sample, ensure_ascii=False))

    with open(json_file, 'w', encoding='utf-8') as f:
        for text in all_texts:
            f.write("%s\n" % text)


if __name__ == '__main__':
    # Clinton
    file = "Clinton_train.csv"
    json_file = "Clinton_train.json"
    read_gcdt_csv_to_json(file, json_file)

    file = "Clinton_test.csv"
    json_file = "Clinton_test.json"
    read_gcdt_csv_to_json(file, json_file)

    # Enron
    file = "Enron_train.csv"
    json_file = "Enron_train.json"
    read_gcdt_csv_to_json(file, json_file)

    file = "Enron_test.csv"
    json_file = "Enron_test.json"
    read_gcdt_csv_to_json(file, json_file)

    # Yahoo
    file = "Yahoo_train.csv"
    json_file = "Yahoo_train.json"
    read_gcdt_csv_to_json(file, json_file)

    file = "Yahoo_test.csv"
    json_file = "Yahoo_test.json"
    read_gcdt_csv_to_json(file, json_file)

    # Yelp
    file = "Yelp_train.csv"
    json_file = "Yelp_train.json"
    read_gcdt_csv_to_json(file, json_file)

    file = "Yelp_test.csv"
    json_file = "Yelp_test.json"
    read_gcdt_csv_to_json(file, json_file)
