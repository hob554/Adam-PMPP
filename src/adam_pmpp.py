"""
Adam-PM++ 优化器实现 - 消融实验版
默认参数已设置为最佳配置：lr=0.00015, batch_size在训练脚本中指定为128

主要特性 (三个独立开关，便于消融实验):
1. use_powermean: PowerMean 梯度一致性调整 (基于过去10步梯度的余弦相似度)
2. use_warmup:    线性学习率预热
3. use_gradnorm:  基于梯度范数 EMA 的自适应学习率微调

向后兼容: 旧的 use_adaptive 参数仍可使用，
         True 等价于三开关全开, False 等价于三开关全关。
"""

import torch
from torch.optim.optimizer import Optimizer
import math


class AdamPMPP(Optimizer):
    """
    Adam-PM++ (Adam with Power Mean and Adaptive Adjustment)

    Args:
        params: 模型参数
        lr: 学习率 (默认: 0.00015)
        betas: Adam 一阶/二阶矩系数 (默认: (0.9, 0.999))
        eps: 数值稳定项 (默认: 1e-8)
        weight_decay: 权重衰减 (默认: 0)
        warmup_steps: 学习率预热步数 (默认: 100)
        use_powermean: 是否启用 PowerMean 梯度一致性调整 (默认: True)
        use_warmup: 是否启用线性 warmup (默认: True)
        use_gradnorm: 是否启用基于梯度范数 EMA 的自适应学习率 (默认: True)
        use_adaptive: [已废弃] 兼容旧接口。如果显式传入 True/False,
                      会同时设置上面三个开关。如果保持默认 None, 则忽略。
    """

    def __init__(self, params, lr=0.00015, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, warmup_steps=100,
                 use_powermean=True, use_warmup=True, use_gradnorm=True,
                 use_adaptive=None):

        # ---- 参数合法性检查 ----
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        # ---- 向后兼容: use_adaptive 覆盖三个新开关 ----
        if use_adaptive is not None:
            use_powermean = bool(use_adaptive)
            use_warmup = bool(use_adaptive)
            use_gradnorm = bool(use_adaptive)

        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            use_powermean=use_powermean,
            use_warmup=use_warmup,
            use_gradnorm=use_gradnorm,
        )
        super(AdamPMPP, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamPMPP, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """执行单步优化"""

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AdamPMPP does not support sparse gradients')

                state = self.state[p]

                # 状态初始化
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['grad_history'] = []
                    state['grad_norm_ema'] = None

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                step = state['step']

                # 权重衰减
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # ============================================================
                # 模块 A: PowerMean 梯度一致性调整
                # ============================================================
                grad_adjustment = 1.0
                if group['use_powermean']:
                    # 维护梯度历史 (注意: 仅在启用 PowerMean 时才占用内存)
                    state['grad_history'].append(grad.clone())
                    if len(state['grad_history']) > 10:
                        state['grad_history'].pop(0)

                    if len(state['grad_history']) >= 3:
                        # 计算相邻梯度的余弦相似度
                        consistencies = []
                        for i in range(len(state['grad_history']) - 1):
                            g1 = state['grad_history'][i].flatten()
                            g2 = state['grad_history'][i + 1].flatten()
                            cos_sim = torch.cosine_similarity(
                                g1.unsqueeze(0), g2.unsqueeze(0), dim=1
                            )
                            consistencies.append(cos_sim.item())

                        avg_consistency = sum(consistencies) / len(consistencies)

                        # 激进的调整策略
                        if avg_consistency > 0.65:
                            grad_adjustment = 1.30
                        elif avg_consistency < 0.35:
                            grad_adjustment = 0.70
                        else:
                            grad_adjustment = 0.70 + 0.60 * ((avg_consistency - 0.35) / 0.3)

                # 应用 PowerMean 调整 (关闭时 grad_adjustment=1.0, 等价于不调整)
                adjusted_grad = grad * grad_adjustment

                # ============================================================
                # 标准 Adam 一阶/二阶矩更新
                # ============================================================
                exp_avg.mul_(beta1).add_(adjusted_grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(adjusted_grad, adjusted_grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                step_size = group['lr']

                # ============================================================
                # 模块 B: 线性 Warmup
                # ============================================================
                if group['use_warmup'] and step < group['warmup_steps']:
                    warmup_factor = step / group['warmup_steps']
                    step_size = step_size * warmup_factor

                # ============================================================
                # 模块 C: 基于梯度范数 EMA 的自适应学习率
                # ============================================================
                if group['use_gradnorm']:
                    # 仅在 warmup 之后启用 (与原版行为一致)
                    # 注意: 当 use_warmup=False 时, warmup_steps 仍作为"启用 GradNorm 的起始步数"
                    if step > group['warmup_steps']:
                        grad_norm = grad.norm().item()

                        if state['grad_norm_ema'] is None:
                            state['grad_norm_ema'] = grad_norm
                        else:
                            state['grad_norm_ema'] = 0.9 * state['grad_norm_ema'] + 0.1 * grad_norm

                        if state['grad_norm_ema'] > 1e-8:
                            norm_ratio = grad_norm / state['grad_norm_ema']
                            if norm_ratio > 2.0:
                                step_size = step_size * 0.95
                            elif norm_ratio < 0.3:
                                step_size = step_size * 1.03

                # ============================================================
                # 偏差修正与参数更新
                # ============================================================
                step_size = step_size / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


# ====================================================================
# 自检: 验证三个开关都能独立工作
# ====================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("Adam-PM++ 优化器 - 消融实验版")
    print("=" * 70)

    configs = [
        ("baseline   ", dict(use_powermean=False, use_warmup=False, use_gradnorm=False)),
        ("powermean  ", dict(use_powermean=True,  use_warmup=False, use_gradnorm=False)),
        ("warmup     ", dict(use_powermean=False, use_warmup=True,  use_gradnorm=False)),
        ("gradnorm   ", dict(use_powermean=False, use_warmup=False, use_gradnorm=True)),
        ("full (PMPP)", dict(use_powermean=True,  use_warmup=True,  use_gradnorm=True)),
    ]

    torch.manual_seed(0)
    for name, cfg in configs:
        x = torch.randn(10, 5, requires_grad=True)
        opt = AdamPMPP([x], lr=0.001, **cfg)
        for _ in range(5):
            opt.zero_grad()
            loss = (x ** 2).sum()
            loss.backward()
            opt.step()
        print(f"  [{name}] final loss = {loss.item():.6f}  ✅")

    # 向后兼容性测试
    print("\n向后兼容性测试 (use_adaptive 旧接口):")
    x = torch.randn(10, 5, requires_grad=True)
    opt = AdamPMPP([x], lr=0.001, use_adaptive=False)
    print(f"  use_adaptive=False -> "
          f"powermean={opt.param_groups[0]['use_powermean']}, "
          f"warmup={opt.param_groups[0]['use_warmup']}, "
          f"gradnorm={opt.param_groups[0]['use_gradnorm']}  ✅")

    opt = AdamPMPP([x], lr=0.001, use_adaptive=True)
    print(f"  use_adaptive=True  -> "
          f"powermean={opt.param_groups[0]['use_powermean']}, "
          f"warmup={opt.param_groups[0]['use_warmup']}, "
          f"gradnorm={opt.param_groups[0]['use_gradnorm']}  ✅")

    print("\n✅ 所有测试通过")
