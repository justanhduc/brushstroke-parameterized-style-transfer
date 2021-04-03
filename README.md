# Parametrized-brushstroke Image Style Transfer

This is a Pytorch implementation of the paper 
["Rethinking Style Transfer: From Pixels to Parameterized Brushstrokes"](http://arxiv.org/abs/2103.17185).
At the moment, due to missing details and the 
[official code](https://github.com/CompVis/brushstroke-parameterized-style-transfer) 
not yet available, this implementation only serves as a proof-of-concept. 
The algorithm will be modified once the official code is available.

## Prerequisites

[Neuralnet-pytorch](https://github.com/justanhduc/neuralnet-pytorch) (For logging)

## Running the code

```
python nst.py
```

## Differences from the paper

- The visual quality is not good as images suffer from blocking artifacts 
due to the nearest neighbor upsampling in Section C2 in the supplementary.
- The stroke mask is implemented without the 2-norm, 
  as the mask formula seems to be wrong.
- The logit of the softmax is negative in the assignment tensor.
- The learning rate is `1e-3`.
- The optimization runs for 20,000 steps.
- The weights of loss terms probably are different, 
  as they are not mentioned in the paper
  
## Results

(TBU)
