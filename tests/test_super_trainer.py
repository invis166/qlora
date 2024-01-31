import torch
from torch import nn
from unittest.mock import Mock, patch
import pytest
from trainer.super_trainer import SuperTrainer


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_1 = nn.Parameter(torch.tensor([[0, 0], [0, 1]], dtype=torch.float32))
        self.lora_2 = nn.Parameter(torch.tensor([[0, 0], [1, 0]], dtype=torch.float32))
        self.lora_3 = nn.Parameter(torch.tensor([[0, 1], [0, 0]], dtype=torch.float32))
        self.lora_4 = nn.Parameter(torch.tensor([[1, 0], [0, 0]], dtype=torch.float32))
        self.dummy = nn.Parameter(torch.tensor(1, dtype=torch.float32))


@pytest.fixture()
def model():
    model = MockModel()

    return model


@pytest.fixture()
def mock_training_step():
    with patch('transformers.Trainer.training_step') as training_step_mock:
        yield training_step_mock


def do_training_step(trainer, model, input):
    trainer.training_step(model, input)
    trainer.state.global_step += 1


def get_lora_adapters(model: MockModel):
    return {name: param for name, param in model.named_parameters() if 'lora_' in name}


def test_calls_transformers_training_step_when_calling_training_step(model: MockModel, mock_training_step):
    x = {'input_ids': [1]}
    trainer = SuperTrainer(model=model, warmup_steps=1, sparse_training_steps=1, sparsity=1)
    do_training_step(trainer, model, x)

    mock_training_step.assert_called_once_with(model, x)


def test_saves_lora_adapters_at_first_warmup_run(model: MockModel, mock_training_step):
    x = {'input_ids': [1]}
    trainer = SuperTrainer(model=model, warmup_steps=2, sparse_training_steps=1, sparsity=1)
    do_training_step(trainer, model, x)

    lora_adapters = get_lora_adapters(model)
    assert trainer._adapters.keys() == lora_adapters.keys()
    for name, saved_adapter in trainer._adapters.items():
        expected_adapter = lora_adapters[name]
        assert torch.allclose(saved_adapter, expected_adapter)


@pytest.mark.parametrize(
    'sparsity', [0.25, 0.5, 0.75, 1]
)
def test_proportion_of_freezed_adapters_at_warmup_end(model: MockModel, mock_training_step, sparsity):
    x = {'input_ids': [1]}
    warmup_steps = 2
    trainer = SuperTrainer(model=model, warmup_steps=warmup_steps, sparse_training_steps=1, sparsity=sparsity)

    for _ in range(warmup_steps + 1):
        do_training_step(trainer, model, x)

    lora_adapters = get_lora_adapters(model)
    expected_num_freezed = int(sparsity * len(lora_adapters))
    actual_num_freezed = len([param for param in model.parameters() if not param.requires_grad])
    assert actual_num_freezed == expected_num_freezed


def test_frees_saved_adapters_at_warmup_end(model: MockModel, mock_training_step):
    x = {'input_ids': [1]}
    trainer = SuperTrainer(model=model, warmup_steps=1, sparse_training_steps=1, sparsity=0.5)

    do_training_step(trainer, model, x)
    do_training_step(trainer, model, x)

    assert not trainer._adapters


def test_freezed_least_changed_at_wamup_end(model: MockModel, mock_training_step):
    x = {'input_ids': [1]}
    trainer = SuperTrainer(model=model, warmup_steps=1, sparse_training_steps=1, sparsity=0.5)
    model.to(torch.device('cpu'))

    delta = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32)
    do_training_step(trainer, model, x)
    model.lora_1.data += delta
    model.lora_3.data += delta
    do_training_step(trainer, model, x)

    freezed_layers = [name for name, param in model.named_parameters() if not param.requires_grad]

    assert len(freezed_layers) == 2
    assert 'lora_2' in freezed_layers
    assert 'lora_4' in freezed_layers


def test_unfreezes_all_adapters_at_sparse_training_end(model: MockModel, mock_training_step):
    x = {'input_ids': [1]}
    trainer = SuperTrainer(model=model, warmup_steps=1, sparse_training_steps=1, sparsity=0.5)

    for _ in range(3):
        do_training_step(trainer, model, x)

    for name, parameter in model.named_parameters():
        assert parameter.requires_grad
