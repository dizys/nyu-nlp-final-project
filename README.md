# nyu-nlp-final-project

NYU NLP Final Project: Build a semantic role labeling system utilizing SOTA machine learning models

## Authors

- [Ziyang Zeng](https://github.com/dizys): Maxent Baseline, DistilBERT related experiments
- [Jiahao Chen](https://github.com/jc10347): Random Forest
- [Peiwen Tang](https://github.com/ppppppw): Feature engineering, Word2Vec and downstream experiments
- [Zeyu Yang](https://github.com/MalikYang9636): BERT base model

## Experiments

- Baseline: [Maxent](https://github.com/dizys/nyu-nlp-homework-6)
- Word2Vec: [Feature Extraction](./feature_extraction/word2vec.py) and [Classification](./feature_extraction/hf_transformer_word2vec.ipynb)
- BERT (base): [Notebook](./src/bert_base.ipynb)
- DistilBERT: [Notebook](./src/hf_transformer.ipynb)
- DistilBERT (POS+BIO): [Notebook](./src/hf_transformer_enhanced.ipynb)
- DistilBERT (QA): [Notebook](./src/hf_transformer_qa.ipynb)
- DistilBERT (ONE ARG1): [Notebook](./src/hf_transformer_one_arg.ipynb)

## Results

| Model                     | Precision | Recall |     F1      | Output                                        |
| :------------------------ | :-------: | :----: | :---------: | :-------------------------------------------- |
| Maxent                    |   71.88   | 61.33  |    66.19    | [txt](./out/test-out-maxent.txt)              |
| RandomForest              |   64.53   | 74.00  |    68.94    | [txt](./out/test-out-rf.txt)                  |
| WORD2VEC + Bi-LSTM        |     -     |   -    | in progress | [txt](./out/test-word2vec.txt)                |
| BERT                      |   91.33   | 91.33  |    91.33    | [txt](./out/test-out-bert.txt)                |
| DistilBERT                |   93.75   | 90.00  |    91.84    | [txt](./out/test-out-distilbert.txt)          |
| DistilBERT (POS+BIO)      |   93.19   | 91.33  |    92.25    | [txt](./out/test-out-distilbert-enhanced.txt) |
| DistilBERT (QA)           |   92.00   | 92.00  |    92.00    | [txt](./out/test-out-distilbert-qa.txt)       |
| **DistilBERT (ONE ARG1)** |   92.67   | 92.67  |  **92.67**  | [txt](./out/test-out-distilbert-one-arg1.txt) |
