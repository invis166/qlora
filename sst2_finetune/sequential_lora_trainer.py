from typing import Dict, Union, Any

import torch
from torch import nn
from transformers import Trainer


class SequentialLoraTrainer(Trainer):
    """A trainer that trains k-th lora layer on every t-th step (i.e. when layer_idx % step_num == t)"""

    def __init__(self, *args, t, **kwargs):
        super().__init__(*args, **kwargs)

        self.t = t
        self.last_train_step = -1  # a value that will not be equal to an any valid step number

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        self._freeze_lora_matrices_based_on_current_step()

        return super().training_step(model, inputs)

    def _freeze_lora_matrices_based_on_current_step(self):
        if self.state.global_step == self.last_train_step:
            return
        # we need to track a train step change because of the gradient accumulation

        self._freeze_all_lora_matrices()
        for i, layer in enumerate(self.model.layers):
            if i % self.t == ((self.state.global_step + 1) % self.t):
                self._unfreeze_layer_lora_matrices(layer)

        self.last_train_step = self.state.global_step

    def _freeze_all_lora_matrices(self):
        for parameter in self.lora_layers:
            parameter.requires_grad = False

    def _unfreeze_layer_lora_matrices(self, layer):
        for parameter_name, parameter in layer.named_parameters():
            if 'lora_' in parameter_name:
                parameter.requires_grad = True
