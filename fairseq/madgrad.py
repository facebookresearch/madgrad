

class MADGRAD(torch.optim.Optimizer):
    """
        Adagrad dual averaging form with iterate averaging momentum.
        Lamb scales in this version, meaning that a LR schedule must be used.
    """

    def __init__(self, params, lr=1e-2, momentum=0, weight_decay=0, eps=1e-6):
        defaults = dict(
            lr=lr, eps=madgrad_eps, momentum=momentum, 
            weight_decay=weight_decay, k=0)
        super(MADGRAD, self).__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self):
        return False

    @property
    def supports_flat_params(self):
        return True
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            eps = group['eps']
            k = group['k']
            lr = group['lr'] + eps
            decay = group['weight_decay']
            momentum = group['momentum']

            ck = 1 - momentum
            lamb = lr * math.pow(k+1, 0.5)

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if 'grad_sum_sq' not in state:
                    state['grad_sum_sq'] = torch.zeros_like(p.data).detach()
                    
                if 's' not in state:
                    state['s'] = torch.zeros_like(p).detach()
                    
                if 'x0' not in state:
                    state['x0'] = torch.clone(p.data).detach()
                    
                grad_sum_sq = state['grad_sum_sq']
                s = state['s']
                x0 = state['x0']

                # Apply weight decay
                if decay != 0:
                    grad.add_(p.data, alpha=decay)

                # Accumuate second moments
                grad_sum_sq.addcmul_(grad, grad, value=lamb)
                rms = grad_sum_sq.pow(1/3).add_(eps)

                # s update
                s.data.add_(grad, alpha=lamb)

                # z iterate
                z = x0.addcdiv(s, rms, value=-1)

                # x is a moving average of z
                p.data.mul_(1-ck).add_(z, alpha=ck)

            group['k'] = group['k'] + 1
        return loss
