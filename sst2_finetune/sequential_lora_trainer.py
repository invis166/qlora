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

        decoder_layers = self.model.base_model.model.model.layers
        for i, layer in enumerate(decoder_layers):
            # if i % self.t == ((self.state.global_step + 1) % self.t): -- use this if you want to freeze every t-th layer
            if i <= len(decoder_layers) // 2 - 1 and self.state.global_step % 2 == 0:
                self._unfreeze_layer_lora_matrices(layer)
            elif i > len(decoder_layers) // 2 - 1 and self.state.global_step % 2 == 1:
                self._unfreeze_layer_lora_matrices(layer)
            else:
                self._freeze_layer_lora_matrices(layer)

        self.last_train_step = self.state.global_step

    def _freeze_layer_lora_matrices(self, layer):
        for parameter_name, parameter in layer.named_parameters():
            if 'lora_' in parameter_name:
                parameter.requires_grad = False

    def _unfreeze_layer_lora_matrices(self, layer):
        for parameter_name, parameter in layer.named_parameters():
            if 'lora_' in parameter_name:
                parameter.requires_grad = True
