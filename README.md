# NonFactS
NonFactS: Nonfactual Summary Generation for Factuality Evaluation in Document Summarization (accepted at ACL2023)

**Authors:** Amir Soleimani, Christof Monz, Marcel Worring

## Abstract
Pre-trained abstractive summarization models can generate fluent summaries and achieve high ROUGE scores. Previous research has found that these models often generate summaries that are inconsistent with their context document and contain nonfactual information. To evaluate factuality in document summarization, a document-level Natural Language Inference (NLI) classifier can be used. However, training such a classifier requires large-scale high-quality factual and nonfactual samples. To that end, we introduce NonFactS, a data generation model, to synthesize nonfactual summaries given a context document and a human-annotated (reference) factual summary. Compared to previous methods, our nonfactual samples are more abstractive and more similar to their corresponding factual samples, resulting in state-of-the-art performance on two factuality evaluation benchmarks, FALSESUM and SUMMAC. Our experiments demonstrate that even without human-annotated summaries, NonFactS can use random sentences to generate nonfactual summaries and a classifier trained on these samples generalizes to out-of-domain documents.

## Limitations
NonFactS generates grammatically correct nonfactual summaries. However, in practice, summaries can be non-grammatical, noisy, and nonsensical. This can limit the generalization of our performance in such cases. Additionally, hypothesis-only results show that a considerable number of samples are identified correctly without their context document. The reason can be the memorized knowledge in pre-trained classifiers or surface features and semantic plausibility.

## Broader Impact
Our model has no direct environmental impacts, fairness or privacy considerations. However, it is important to note that it must not be used as a fact-checking tool as there is a potential risk that false statements may be labelled as true. Our classifier evaluates the factuality of a summary based on a context document, and if the document is misleading, the summary can be factual based on misleading information. Additionally, NonFactS generates nonfactual summaries, which might have potential risks if misused for generating massive nonfactual summaries (claims). Addressing such risks is an open issue in the field and is not specific to our work.



# Requirements

## Installation
- Create conda environment
  
        conda create -n NonFactS python=3.6
        conda activate NonFactS
        pip install -r requirements.txt

- Install pytorch 1.7.1 (according to your cuda & gpus)

        conda install pytorch torchvision torchaudio cudatoolkit=9.2 -c pytorch

Note: double check if Transformers (huggingface) version is (4.4.0.dev0)


### Training datasets (Factual and NonFactual summaries)
Training datasets contain 50% positive (Factual) summaries and 50% negative (NonFactual) summaries

- 100k samples (FALSESUM benchmark) (100k.csv) \
https://drive.google.com/file/d/1_1nSMyMH7pW37OryzsaehO9CurZ3ZKPt/view?usp=share_link

- 100k samples + MNLI (FALSESUM benchmark) (100k_MNLI.csv) \
https://drive.google.com/file/d/15T2mmr0s8P5DCIof4x5ZGv1szXtb4sxE/view?usp=sharing

- 200k samples (SUMMAC benchmark) (200k.csv) \
  https://drive.google.com/file/d/1TsmvwRyvG7Kfdy3LtL6lbyStfAy8q0iJ/view?usp=share_link

### Models 

#### Classifier:

- Roberta-base trained on 100k samples (FALSESUM benchmark):\
  https://drive.google.com/file/d/1WVyuUVk23hRdhxzz6Q1ZoY6w6OXZflVF/view?usp=share_link
- Roberta-base trained on 100k samples + MNLI (FALSESUM benchmark): \
  https://drive.google.com/file/d/1K3gyCeGsp0OOqjFFH_2ybPti_Vp2FteV/view?usp=share_link
- ALBERT-xlarge trained on 200k samples (SUMMAC benchmark) \
https://drive.google.com/file/d/1ncxGb_6hM27hiv2Ra5921HogmlUsrHYq/view?usp=share_link
- ALBERT-xxlarge trained on 200k samples (SUMMAC benchmark)
https://drive.google.com/file/d/1UPexanNjS6BqO1nwhUePId2TFvzWFlvX/view?usp=share_link

 
      python -u  run_classifier.py \
      --model_name_or_path roberta-base \
      --do_train \
      --do_eval \
      --do_predict \
      --max_seq_length 512 \
      --per_device_train_batch_size 8 \
      --gradient_accumulation_steps 2 \
      --learning_rate 1e-5 \
      --num_train_epochs 1 \
      --output_dir output_path \
      --overwrite_output_dir \
      --train_file train_file \
      --validation_file factcc.csv \
      --use_fast_tokenizer False \
      --save_strategy "no" \
      --warmup_ratio 0.06 \
      --weight_decay 0.1 \
      --logging_steps=100 \
      --report_to none \

#### Generator:

Download the training and test dataset: 
- 50k summaries for inference: \
https://drive.google.com/file/d/1Y3B7ZnNVN8OH8RjWKgl3nMldDHEmUyZI/view?usp=share_link
- rest of summaries for training: \
https://drive.google.com/file/d/15W4aXoDdOhN3EKxuFXTXsZxZt26HSfnw/view?usp=share_link 
- download the trained model if you just want to do inference (generating nonfactual summaries): \
https://drive.google.com/file/d/16uhiU3BRlbQYJnBnqpglrg0sv51N0AKN/view?usp=share_link


- **Training:** training a BART-base model (see figure 2 in the paper) 
 

      python -u train_seq2seq.py \
      --model_name_or_path facebook/bart-base \
      --text_column doc \
      --summary_column summary \
      --do_train \
      --do_predict \
      --task summarization \
      --train_file cnndm_sentence_50000_rest.csv \
      --validation_file cnndm_sentence_50000_firstsum.csv \
      --test_file cnndm_sentence_50000_firstsum.csv \
      --output_dir output_path  \
      --per_device_train_batch_size=2 \
      --per_device_eval_batch_size=16 \
      --overwrite_output_dir \
      --predict_with_generate \
      --gradient_accumulation_steps 10 \
      --num_train_epochs 1 \
      --save_strategy "no" \
      --learning_rate=3e-05 \
      --weight_decay=0.01 \
      --max_grad_norm=0.1 \
      --lr_scheduler_type=polynomial \
      --warmup_steps=500 \
      --label_smoothing_factor=0.1 \
      --config_name my_config2.json \
      --logging_steps=100 \
      --max_source_length=1024 \
      --report_to none \


- **Inference:** testing the trained BART-base model to generate nonfactual summaries (see figure 2 in the paper)

    
      python -u inference_seq2seq.py \
      --model_name_or_path o_train_bart_cnn_percent50_stopwords_sep_halfsum_plus4timesrand_e1 \
      --text_column doc \
      --summary_column summary \
      --do_predict \
      --task summarization \
      --train_file cnndm_sentence_50000_firstsum.csv \
      --validation_file cnndm_sentence_50000_firstsum.csv \
      --test_file cnndm_sentence_50000_firstsum.csv \
      --output_dir output_path \
      --per_device_train_batch_size=2 \
      --per_device_eval_batch_size=32 \
      --overwrite_output_dir \
      --predict_with_generate \
      --gradient_accumulation_steps 5 \
      --num_train_epochs 2 \
      --save_strategy "no" \
      --learning_rate=3e-05 \
      --weight_decay=0.01 \
      --max_grad_norm=0.1 \
      --lr_scheduler_type=polynomial \
      --warmup_steps=500 \
      --label_smoothing_factor=0.1 \
      --config_name my_config2.json \
      --logging_steps=100 \
      --max_source_length=1024 \
      --report_to none \
      --max_val_samples=50000 \
      --max_test_samples=50000 \

