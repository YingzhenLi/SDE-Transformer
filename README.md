# SDE-Transformer
TL; DR: Linear SDE version of Attention blocks applied to Transformers

(Yingzhen's exploration during Christmas holidays in Dec 2023.)

## Testing Run
```
python exp.py --data=cifar10 --n_block=4 --dim_emb=128 --block_type=mhsde --epochs=100 --save=True
```
In particular, the ```--block_type``` option support the following models:

``` nn ```: just simple MLP.

```mfvi```: simple MLP-based BNN with mean-field variational inference.

```attention```: multi-head self-attention, when using this option it results in the original ViT structure.

```sgpa```: sparse GP attention (Chen and Li, ICLR 2023).

```sde```: Linear SDE with diffusion-matching stochastic q process approximation learned using VI.

```sparsecorrsde```: Linear SDE with inducing-point based sparse GP approximation (but computed with Kalman filtering/smoothing).

```mhsde```: same as ```sparsecorrsde``` but multi-head of multiple 1D linear SDEs.

For other configs available see ```exp.py```.

## Key References
[Linear SDE: Ornstein–Uhlenbeck process (Wikipedia)](https://en.wikipedia.org/wiki/Ornstein–Uhlenbeck_process)

[Archambeau et al., NIPS 2007 "Variational Inference for Diffusion Processes"](https://papers.nips.cc/paper_files/paper/2007/hash/818f4654ed39a1c147d1e51a00ffb4cb-Abstract.html)

[Adam et al., AISTATS 2020 "Doubly Sparse Variational Gaussian Processes"](http://proceedings.mlr.press/v108/adam20a.html)

[Chen and Li, ICLR 2023 "Calibrating Transformers via Sparse Gaussian Processes"](https://openreview.net/forum?id=jPVAFXHlbL)
