# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import types

import torch
import torch.optim
import torch.distributed as dist
import copy
import math
import pdb
from .madgrad import MADGRAD

try:
    import fairseq
    from fairseq.optim import FairseqOptimizer, register_optimizer
except ImportError:
    _has_fairseq = False
else:
    _has_fairseq = True

if _has_fairseq:
    @register_optimizer('madgrad')
    class FairseqMADGRAD(FairseqOptimizer):
        """
        MADGRAD_: A Momentumized, Adaptive, Dual Averaged Gradient Method for Stochastic 
        Optimization.

        .. _MADGRAD: https://arxiv.org/abs/2101.11075

        MADGRAD is a general purpose optimizer that can be used in place of SGD or
        Adam may converge faster and generalize better. Currently GPU-only.
        Typically, the same learning rate schedule that is used for SGD or Adam may
        be used. The overall learning rate is not comparable to either method and
        should be determined by a hyper-parameter sweep.

        MADGRAD requires less weight decay than other methods, often as little as
        zero. Momentum values used for SGD or Adam's beta1 should work here also.

        On sparse problems both weight_decay and momentum should be set to 0.

        Arguments:
            params (iterable): 
                Iterable of parameters to optimize or dicts defining parameter groups.
            lr (float): 
                Learning rate (default: 1e-2).
            momentum (float): 
                Momentum value in  the range [0,1) (default: 0.9).
            weight_decay (float): 
                Weight decay, i.e. a L2 penalty (default: 0).
            madgrad_eps (float): 
                Term added to the denominator outside of the root operation to improve numerical stability. (default: 1e-6).
        """

        def __init__(self, args, params):
            super().__init__(args)
            self._optimizer = MADGRAD(params, **self.optimizer_config)

        @staticmethod
        def add_args(parser):
            """Add optimizer-specific arguments to the parser."""
            # fmt: off
            parser.add_argument('--weight-decay', '--wd', default=0.0, type=float, metavar='WD',
                                help='weight decay')
            parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                                help='momentum factor')
            parser.add_argument('--madgrad_eps', default=1e-6, type=float, metavar='M',
                                help='Denominator epsilon')
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
                'eps': self.args.madgrad_eps,
            }
