import inspect
from operator import itemgetter
from enum import Enum
from typing import Dict, Union
from functools import partial

import torch
import transformers
import bitsandbytes as bnb
from torch import nn
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments


class BytesUnit(Enum):
    B = 1
    KB = 1024
    MB = 1024 ** 2
    GB = 1024 ** 3


class CommunicationLoggerCallback(transformers.TrainerCallback):
    def __init__(self, trainer: transformers.Trainer,
                accumulate_on_server: bool = False, log_per_iteration: bool = True, log_total: bool = True,
                unit: Union[BytesUnit, str] = BytesUnit.MB, log_prefix: str = 'communication') -> None:
        """
            trainer: trainer to log values to

            accumulate_on_server: if True, gradients will be considered being accumulated
                on server and transmitted to client after accumulation only once per step.
                If false, gradients will be considered being transmitted client after
                every substep. Default: False

            unit: in what units to log. Options: B, KB, MB, GB
        """

        self._trainer = trainer
        self.accumulate_on_server = accumulate_on_server

        assert log_per_iteration or log_total, "At least on of log_per_iteration or log_total should be True"

        self.log_per_iteration = log_per_iteration
        self.log_total = log_total

        if isinstance(unit, str):
            unit = BytesUnit[unit]

        self.unit = unit
        self.prefix = log_prefix
        self.total_bytes_grad = 0
        self.total_bytes_weight = 0

        # list of torch.nn classes to check if layer is known
        self._torch_layers = set(map(itemgetter(1), inspect.getmembers(torch.nn, inspect.isclass)))

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        """
            Logs size of transmitted data to trainer. Currently counts only gradients by weights, transmitted to client
        """

        if state.is_local_process_zero:
            # logging gradients transmittion
            grad_size = self.get_gradients_size(kwargs['model'])
            weights_size = grad_size

            if self.accumulate_on_server:
                grad_size *= args.gradient_accumulation_steps

            self.total_bytes_grad += grad_size
            self.total_bytes_weight += weights_size

            to_log = {}
            add_metric = partial(self._add_metric_to_log, to_log)

            if self.log_per_iteration:
                add_metric('grad_size', grad_size)
                add_metric('weights_size', weights_size)

            if self.log_total:
                add_metric('total_grad_size', self.total_bytes_grad)
                add_metric('total_weights_size', self.total_bytes_weight)
                add_metric('total_transmitted', self.total_bytes_weight + self.total_bytes_grad)

            self._trainer.log(to_log)
            control.should_log = True

    def _add_metric_to_log(self, to_log: Dict[str, float], metric: str, value: Union[float, int]) -> None:
        name = self.prefix + '/' + metric + '_' + self.unit.name
        to_log[name] = value / self.unit.value

    def get_tensor_element_size(self, tensor: torch.Tensor) -> int:
        """
        Returns size of one element in tensor in bits (!), not bytes
        """

        if isinstance(tensor, bnb.nn.Params4bit):
            return 4
        elif isinstance(tensor, bnb.nn.Int8Params):
            return 8
        elif tensor in self._torch_layers:
            return 8 * tensor.element_size()
        else:
            raise ValueError(f"Unknown tensor type: \"{type(tensor)}\"")

    def get_tensor_size(self, tensor: torch.Tensor) -> float:
        """
        Returns tensor size in bytes
        """

        return (tensor.numel() * self.get_tensor_element_size(tensor)) / 8.0

    def get_model_size(self, model: nn.Module) -> float:
        """
        Returns total model size in bytes
        """

        total = 0.0

        for param in model.parameters():
            total += self.get_tensor_size(param)

        return total

    @classmethod
    def get_gradients_size(cls, model: nn.Module) -> int:
        """
        Returns total model gradients size in bytes.
        """

        total_bytes = 0

        for param in model.parameters():
            if param.requires_grad:
                total_bytes += param.numel() * param.element_size()

        return total_bytes
