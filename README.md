# 自然语言处理-命名实体识别

对比常见命名实体识别模型效果，主要涉及以下几种模型：

- [Fast and Accurate Entity Recognition with Iterated Dilated Convolutions](https://arxiv.org/pdf/1702.02098.pdf)
- [CAN-NER: Convolutional Attention Network for Chinese Named Entity Recognition](https://arxiv.org/pdf/1904.02141.pdf)
- [GRN: Gated Relation Network to Enhance Convolutional Neural Network for Named Entity Recognition](https://arxiv.org/pdf/1907.05611.pdf)
- [TENER: Adapting Transformer Encoder for Named Entity Recognition](https://arxiv.org/pdf/1911.04474.pdf)
- [Enhancing Entity Boundary Detection for Better Chinese Named Entity Recognition](https://aclanthology.org/2021.acl-short.4.pdf)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

## 效果

### ID-CNN

|-|Simple|CRF|Segment|Bigram|Segment + Bigram|CRF + Segment|CRF + Bigram|CRF + Segment + Bigram|CRF + Segment + Bigram + Fix Embedding|
|----|----|----|----|----|----|----|----|----|----|
|MSRA|0.621|0.664|0.705|<b>0.827</b>|0.821|0.751|0.817|0.797|0.683|
|Weibo|0.226|0.12|0.225|0.33|<b>0.367</b>|0.18|0.347|0.305|0.295|
|Resume|0.853|0.855|0.885|0.905|0.889|0.884|0.914|<b>0.921</b>|0.91|

### CAN-NER

|-|Simple|CRF|Segment|Bigram|CRF + Segment|CRF + Bigram|CRF + Segment + Bigram|CRF + Segment + Bigram + RNN|CRF + Segment + Bigram + Fix Embedding|CRF + Segment + Bigram + RNN + Fix Embedding|
|----|----|----|----|----|----|----|----|----|----|----|
|MSRA|0.643|0.861|0.765|0.809|0.824|<b>0.902</b>|0.834|0.839|0.84|0.835|
|Weibo|0.3|0.47|0.277|0.378|0.502|0.49|0.525|0.527|<b>0.528</b>|<b>0.528</b>|
|Resume|0.801|0.938|0.805|0.884|0.932|<b>0.949</b>|0.943|0.937|0.941|0.944|

### GRN

|-|Simple|CRF|Segment|Bigram|Segment + Bigram|CRF + Segment|CRF + Bigram|CRF + Segment + Bigram|CRF + Segment + Bigram + Fix Embedding|
|----|----|----|----|----|----|----|----|----|----|
|MSRA|0.719|0.827|0.775|0.802|0.799|0.79|<b>0.876</b>|0.819|0.822|
|Weibo|0.404|0.484|0.412|0.427|0.444|0.503|0.512|0.536|<b>0.54</b>|
|Resume|0.877|0.936|0.883|0.902|0.903|0.935|0.941|0.942|<b>0.943</b>|

### TENER

|-|Simple|CRF|Segment|Bigram|Segment + Bigram|CRF + Segment|CRF + Bigram|CRF + Segment + Bigram|CRF + Segment + Bigram + Fix Embedding|
|----|----|----|----|----|----|----|----|----|----|
|MSRA|0.67|0.816|0.741|0.795|0.781|0.777|<b>0.876</b>|0.807|0.81|
|Weibo|0.134|0.265|0.154|0.309|0.265|0.339|0|0.235|<b>0.387</b>|
|Resume|0.796|<b>0.928</b>|0.812|0.84|0.849|0.82|0.777|0.891|0.861|

### Star

|-|Simple|Bound|Bound + CRF|Segment + Bound + CRF|Bigram + Bound + CRF|Segment + Bigram + Bound + CRF|Segment + Bigram + Bound + CRF + Fix Embedding|
|----|----|----|----|----|----|----|----|
|MSRA|0.5|0.76|0.828|0.807|<b>0.898</b>|0.828|0.83|
|Weibo|0.055|0.321|0.474|0.49|0.499|0.522|<b>0.528</b>|
|Resume|0.581|0.881|0.933|<b>0.934</b>|0.93|0.933|0.933|

### Bert

|-|Simple|CRF|Fix Embedding|CRF + Fix Embedding|
|----|----|----|----|----|
|MSRA|<b>0.944</b>|0.943|0.757|0.887|
|Weibo|0.667|<b>0.673</b>|0.051|0.309|
|Resume|<b>0.957</b>|<b>0.957</b>|0.783|0.862|
