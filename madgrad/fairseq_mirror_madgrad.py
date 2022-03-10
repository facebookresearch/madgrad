# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.optim
from .mirror_madgrad import MirrorMADGRAD

try:
    import fairseq
    from fairseq.optim import register_optimizer
    try:
        from fairseq.optim import LegacyFairseqOptimizer as FairseqOptimizer 
    except:
        from fairseq.optim import FairseqOptimizer
except ImportError:
    _has_fairseq = False
else:
    _has_fairseq = True

if _has_fairseq:
    @register_optimizer('mirror_madgrad')
    class FairSeqMirrorMADGRAD(FairseqOptimizer):
        """
        Mirror MADGRAD_: A Momentumized, Adaptive, Dual Averaged Gradient Method for Stochastic 
        Optimization.

        .. _MADGRAD: https://arxiv.org/abs/2101.11075

        Mirror MADGRAD uses the weighting and momentum of MADGRAD but uses mirror descent
        rather than dual averaging as the base method. In general, the mirror variant works 
        better than standard MADGRAD on problems where generalization gap is not an issue, 
        such as large Transformer model training. On CIFAR-10/Image-Net and smaller NLP models
        the standard variant should be prefered. The Mirror variant is more numerically stable
        which may help with large model training.
        
        Currently does not support sparse gradients.

        Arguments:
            params (iterable): 
                Iterable of parameters to optimize or dicts defining parameter groups.
            lr (float): 
                Learning rate (default: 1e-2).
            momentum (float): 
                Momentum value in the range [0,1) (default: 0.9).
            weight_decay (float): 
                Weight decay, i.e. a L2 penalty (default: 0).
            eps (float): 
                Term added to the denominator outside of the root operation to improve numerical stability. (default: 0).
                This parameter is less important in MADGRAD than in Adam. A value of 0 will likely give the best results.
            decouple_decay (bool):
                Apply AdamW style decoupled weight decay (EXPERIMENTAL). 
                Application of decay occurs before the step.
        """

        def __init__(self, args, params):
            super().__init__(args)
            self._optimizer = MirrorMADGRAD(params, **self.optimizer_config)

        @staticmethod
        def add_args(parser):
            """Add optimizer-specific arguments to the parser."""
            # fmt: off
            parser.add_argument('--weight-decay', '--wd', default=0.0, type=float, metavar='WD',
                                help='weight decay')
            parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                                help='momentum factor')
            parser.add_argument('--eps', default=0, type=float, metavar='M',
                                help='Denominator epsilon')
            parser.add_argument('--decouple_decay', default=False, type=bool, metavar='M',
                                help='Decouple weight decay (EXPERIMENTAL)')
            # fmt: on

        @property
        def optimizer_config(self):
            """
            Return a kwarg dictionary that will be used to override optimizer
            args stored in checkpoints. This allows us to load a checkpoint and
            resume training using a different set of optimizer args, e.g., with a
            different learning rate.
            """
            return {
                'lr': self.args.lr[0],
                'momentum': self.args.momentum,
                'weight_decay': self.args.weight_decay,
                'eps': self.args.eps,
                'decouple_decay': self.args.decouple_decay,
            }