
# MADGRAD Optimization Method

A Momentumized, Adaptive, Dual Averaged Gradient Method for Stochastic Optimization

``` pip install madgrad ```

Try it out! A best-of-both-worlds optimizer with the generalization performance of SGD and at least as fast convergence as that of Adam, often faster. A drop-in torch.optim implementation `madgrad.MADGRAD` is provided, as well as a FairSeq wrapped instance. For FairSeq, just import madgrad anywhere in your project files and use the `--optimizer madgrad` command line option, together with `--weight-decay`, `--momentum`, and optionally `--madgrad_eps`.

The madgrad.py file containing the optimizer can be directly dropped into any PyTorch project if you don't want to install via pip. If you are using fairseq, you need the acompanying fairseq_madgrad.py file as well.

Documentation availiable at https://madgrad.readthedocs.io/en/latest/.

## Things to note:
 - You may need to use a lower weight decay than you are accustomed to. Often 0.
 - You should do a full learning rate sweep as the optimal learning rate will be different from SGD or Adam. Best LR values we found were 2.5e-4 for 152 layer PreActResNet on CIFAR10, 0.001 for ResNet-50 on ImageNet, 0.025 for IWSLT14 using `transformer_iwslt_de_en` and 0.005 for RoBERTa training on BookWiki using `BERT_BASE`. On NLP models gradient clipping also helped.

# Tech Report

[Adaptivity without Compromise: A Momentumized, Adaptive, Dual Averaged Gradient Method for Stochastic Optimization](https://arxiv.org/abs/2101.11075)

We introduce MADGRAD, a novel optimization method in the family of AdaGrad adaptive gradient methods. MADGRAD shows excellent performance on deep learning optimization problems from multiple fields, including classification and image-to-image tasks in vision, and recurrent and bidirectionally-masked models in natural language processing. For each of these tasks, MADGRAD matches or outperforms both SGD and ADAM in test set performance, even on problems for which adaptive methods normally perform poorly.


```BibTeX
@misc{defazio2021adaptivity,
      title={Adaptivity without Compromise: A Momentumized, Adaptive, Dual Averaged Gradient Method for Stochastic Optimization}, 
      author={Aaron Defazio and Samy Jelassi},
      year={2021},
      eprint={2101.11075},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

# Results

![vision](https://github.com/facebookresearch/madgrad/blob/main/figures/vision.png?raw=true)
![nlp](https://github.com/facebookresearch/madgrad/blob/main/figures/nlp.png?raw=true)

# License

MADGRAD is licensed under the [MIT License](LICENSE).
