# soft-label-RE
This project provides the implementation of distantly supervised relation extraction with (bag-level) soft-label adjustment.

Details of soft-label adjustment can be found [here](http://aclweb.org/anthology/D17-1189). The implementation is based on Tensorflow 1.0.0 and Python 2.7.

# Model Overview
<p align="center"><img width="40%" src="fig.png"/></p>

## Distantly Supervised Relation Extraction

Distant supervision automatically generates training examples by aligning entity mentions in plain text with those in KB and labeling entity pairs with their relations in KB. If there's no relation link between certain entity pair in KB, it will be labeled as negative instance (NA).

## Multi-instance Learning
The automatic labeling by KB inevitably accompanies with wrong labels because the relations of entity pairs might be missing from KBs or mislabeled.
Multi-instances learning (MIL) is proposed to combat the noise. The method divides the training set into multiple bags of entity pairs (shown in the figure above) and labels the bags with the relations of entity pairs in the KB (**bag-level DS label**).
Each bag consists of sentences mentioning both head and tail entities.
Much effort has been made in reducing the influence of noisy sentences within the bag,
including methods based on at-least-one assumption and attention mechanisms over instances. 

##  Bag-level Mislabeling
As shown in the figure above, due to the absence of (Jan Eliasson, Sweden)(Jan Eliasson is a Swedish diplomat.) from the Nationality relation in the KB,the entity pair is mislabeled as NA.

Actually, no matter how we design the attention weight calculation of the sentences in that bag, the bag would be a noisy instance during training.
So we try to solve the problem from a different point of view. Since the bag-level DS label can be wrong, we design a soft-label adjustment on the bag-level DS label to correct the ill-labeled cases.

# Reference
If you find the code and data resources helpful, please cite the following paper:
```
@inproceedings{liu2017soft,
  title={A Soft-label Method for Noise-tolerant Distantly Supervised Relation Extraction},
  author={Liu, Tianyu and Wang, Kexiang and Chang, Baobao and Sui, Zhifang},
  booktitle={Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing},
  pages={1790--1795},
  year={2017}
}
```
