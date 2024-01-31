import torch
from torch import nn
from typing import Union, Dict, Any

from transformers import Trainer


class SuperTrainer(Trainer):
    def __init__(self, *args, warmup_steps: int, sparse_training_steps: int, sparsity: float, **kwargs):
        assert 0 < sparsity <= 1, 'Sparsity should be in (0, 1]'

        super().__init__(*args, **kwargs)

        self.warmup_steps = warmup_steps
        self.sparse_training_steps = sparse_training_steps
        self.sparsity = sparsity

        self._last_global_step = -1
        self._warmup_steps_done = 0
        self._sparse_training_steps_done = 0
        self._is_warmup = True

        self._adapters: dict[str, torch.Tensor] = {}

    @property
    def _is_sparse_training(self):
        return not self._is_warmup

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        # training step computes loss and gradients, it isn't doing an optimizer step
        if self._is_warmup:
            self._warmup_step(model)
        if self._is_sparse_training:
            self._sparse_training_step(model)

        super().training_step(model, inputs)

        self._last_global_step = self.state.global_step

    def _warmup_step(self, model: nn.Module):
        if self._last_global_step == self.state.global_step:
            return
        if self._warmup_steps_done == 0:
            self._save_trainable_params(model)

        if self._warmup_steps_done == self.warmup_steps:
            self._finalize_warmup(model)
        else:
            self._warmup_steps_done += 1

    def _finalize_warmup(self, model: nn.Module):
        '''Freezes lora adapters that had a small change during the warmup stage'''
        layers_change = self._get_layers_change(model)
        num_layers_to_freeze = int(len(layers_change) * self.sparsity)
        layers_to_freeze = {
            layer_name
            for layer_name, _ in
            sorted(layers_change.items(), key=lambda x: x[1])[:num_layers_to_freeze]
        }
        for parameter_name, parameter in model.named_parameters():
            if 'lora_' not in parameter_name:
                continue
            if parameter_name in layers_to_freeze:
                parameter.requires_grad = False
            else:
                parameter.requires_grad = True

        self._adapters = {}
        self._warmup_steps_done = 0
        self._is_warmup = False

    def _get_layers_change(self, model: nn.Module) -> dict[str, float]:
        '''Calculates norm of the change of the adapters'''
        layers_change = {}
        for name, adapter in self._adapters.items():
            difference = model._parameters[name] - adapter
            layers_change[name] = torch.trace(difference @ difference.T)

        return layers_change

    def _save_trainable_params(self, model: nn.Module):
        '''Saves current values of the trainable adapters'''
        for parameter_name, parameter in model.named_parameters():
            if 'lora_' not in parameter_name:
                continue
            # for now saving tensors in the GPU memory
            # it may be reasonable to save them in the RAM or on the hard drive
            self._adapters[parameter_name] = parameter.detach().clone()

    def _sparse_training_step(self, model: nn.Module):
        if self._last_global_step == self.state.global_step:
            return

        if self._sparse_training_steps_done == self.sparse_training_steps:
            self._finalize_sparse_training(model)
        else:
            self._sparse_training_steps_done += 1

    def _finalize_sparse_training(self, model: nn.Module):
        '''Unfreezes all lora adapters'''
        for parameter_name, parameter in model.named_parameters():
            if 'lora_' in parameter_name:
                parameter.requires_grad = True

        self._sparse_training_steps_done = 0
        self._is_warmup = True

