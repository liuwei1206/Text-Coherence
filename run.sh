python3 trainer.py --do_train --do_eval --model_type="Bert" \
                   --model_name_or_path="data/result/checkpoint-8/pytorch_model.bin" \
                   --config_name="data/pretrained_models/bert/config.json" \
                   --vocab_file="data/pretrained_models/bert/vocab.txt" \
                   --doc_type="Document-paragraph" \
                   --para_pooling_type="max" \
                   --sent_pooling_type="CLS" \
                   --doc_pooling_type="CLS"