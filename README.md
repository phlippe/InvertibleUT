# Combining Universal Transformer and Flow-based models

## Experiment 1: Invertible Universal Transformer
This experiment tests the capability of combining invertible neural networks ([iRevNet](https://arxiv.org/abs/1802.07088), [Reversible ResNet](https://arxiv.org/abs/1707.04585)) and the universal transformer. The idea is to get a memory-efficient backpropagation of the UT allowing us to train it on smaller GPUs. 

* [X] First version implementation 
* [X] Translation from UT parameters to invertible UT parameters (taking half of hidden size)
* [ ] Verify current implementation
* [ ] Check parameter setting for attention (might also reduce channel size) 
* [ ] Adding parameter for deciding whether to share the layers or not
