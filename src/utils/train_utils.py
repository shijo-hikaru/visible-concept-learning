import os
import copy
from copy import deepcopy

import pandas as pd
import torch
from torch import nn

def log_result_table(header, results_dict, title=None, save_dir=None, filename='results', logging=None):
    """
    header: list[str]
    results_dict: dict[str, list[float]]
        例:
        {
            "seen": test_acc_seen,
            "unseen": test_acc_unseen,
            "unseen_only": test_acc_unseen_only
        }
    """
    df = pd.DataFrame.from_dict(
        results_dict,
        orient="index",
        columns=header
    )


    if logging:
        if title is not None:
            logging.info(title)
        logging.info("\n" + df.to_string(float_format="%.4f"))
    else:
        if title is not None:
            print(title)
        print("\n" + df.to_string(float_format="%.4f"))

    if save_dir is not None:
        csv_path = os.path.join(save_dir, f"{filename}.csv")
        df.to_csv(csv_path)


#from https://zenn.dev/yuto_mo/articles/53f3823c0c011d
class ModelEmaV2(nn.Module):
    def __init__(self, model, decay=0.9999, device=None):
        super(ModelEmaV2, self).__init__()
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.module.to(device=device)
        self.backup = {}

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

    def apply_shadow(self):
        """Save current model parameters and replace them with EMA parameters."""
        self.backup = {name: param.data.clone() for name, param in self.module.state_dict().items()}
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.module.state_dict()[name])

    def restore(self):
        """Restore the original model parameters."""
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}



class EMA:
    def __init__(self, model: nn.Module, momentum: float = 0.999):
        """
        model: student CLIP model
        momentum: EMA係数 m
        """
        self.momentum = momentum
        self.model = model
        self.ema_model = copy.deepcopy(model)

        # EMA側は勾配を持たない
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self):
        """
        ema = m * ema + (1-m) * student
        """
        for p_ema, p in zip(self.ema_model.parameters(),
                            self.model.parameters()):
            p_ema.data.mul_(self.momentum).add_(
                p.data, alpha=1.0 - self.momentum
            )





