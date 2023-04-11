# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.optim
import math

from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from torch.optim.optimizer import _params_t
else:
    _params_t = Any

class MirrorMADGRAD(torch.optim.Optimizer):
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

    def __init__(
        self, params: _params_t, lr: float = 1e-2, momentum: float = 0.9, 
        weight_decay: float = 0, eps: float = 0, decouple_decay=False,
    ):
        if momentum < 0 or momentum >= 1:
            raise ValueError(f"Momentum {momentum} must be in the range [0,1]")
        if lr < 0:
            raise ValueError(f"Learning rate {lr} must be non-negative")
        if weight_decay < 0:
            raise ValueError(f"Weight decay {weight_decay} must be non-negative")
        if eps < 0:
            raise ValueError(f"Eps must be non-negative")

        defaults = dict(lr=lr, eps=eps, momentum=momentum, 
            weight_decay=weight_decay, decouple_decay=decouple_decay)
        super().__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self) -> bool:
        return True

    @property
    def supports_flat_params(self) -> bool:
        return True

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        # step counter must be stored in state to ensure correct behavior under
        # optimizer sharding
        if 'k' not in self.state:
            self.state['k'] = torch.tensor([0], dtype=torch.long)
        k = self.state['k'].item()

        update_ratio = math.pow(k/(k+1), 1/2)
        lamb = math.pow(k+1, 1/3)

        for group in self.param_groups:
            eps = group["eps"]
            lr = group["lr"]
            if lr != 0.0:
                lr = lr + eps # For stability
            decay = group["weight_decay"]
            momentum = group["momentum"]
            decouple_decay = group.get("decouple_decay", False)

            ck = 1 - momentum

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                state = self.state[p]

                p_data_fp32 = p.data
                if p.data.dtype in {torch.float16, torch.bfloat16}:
                    p_data_fp32 = p_data_fp32.float()


                if "grad_sum_sq" not in state:
                    state["grad_sum_sq"] = torch.zeros_like(p_data_fp32).detach()
                    state["z"] = torch.clone(p_data_fp32).detach()

                if momentum != 0.0 and grad.is_sparse:
                    raise RuntimeError("momentum != 0 is not compatible with sparse gradients")

                grad_sum_sq = state["grad_sum_sq"]
                z = state["z"]

                # Apply weight decay
                if decay != 0:
                    if grad.is_sparse:
                        raise RuntimeError("weight_decay option is not compatible with sparse gradients")

                    if decouple_decay:
                        z.data.add_(z.data, alpha=-lr*decay)
                    else:
                        grad.add_(p_data_fp32, alpha=decay)

                grad_sum_sq.mul_(update_ratio)
                # Accumulate second moments
                grad_sum_sq.addcmul_(grad, grad, value=1)
                rms = grad_sum_sq.pow(1 / 3).add_(eps)

                if eps == 0:
                    rms[rms == 0] = float('inf')

                # Update z
                z.data.addcdiv_(grad, rms, value=-lr*lamb)

                # Step
                p_data_fp32.mul_(1 - ck).add_(z, alpha=ck)

                if p.data.dtype in {torch.float16, torch.bfloat16}:
                    p.data.copy_(p_data_fp32)

        self.state['k'] += 1
        return loss
