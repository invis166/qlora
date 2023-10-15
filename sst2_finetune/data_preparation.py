import torch
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass
import transformers
from datasets import load_dataset


from typing import Dict, Sequence


@dataclass
class DataCollatorForSequenceClassification(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    predict_with_generate: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sentences = [
            f"{self.tokenizer.bos_token}{example['sentence']}{self.tokenizer.eos_token}"
            for example in instances
        ]
        labels = [example['label'] for example in instances]

        # Tokenize
        tokenized_sentences = self.tokenizer(
            sentences,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        # Build the input for classification
        input_ids = [tokenized_sentece['input_ids'] for tokenized_sentece in tokenized_sentences]
        # Apply padding
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        data_dict = {
            'input_ids': input_ids,
            'attention_mask': input_ids.ne(self.tokenizer.pad_token_id),
            'labels': labels
        }

        return data_dict


def _format_dataset(dataset):
    dataset = dataset.map(remove_columns=['idx'])

    return dataset


def make_data_module(tokenizer: transformers.PreTrainedTokenizer, args) -> Dict:
    """
    Make dataset and collator for supervised fine-tuning.
    """

    # Load dataset.
    dataset = load_dataset(
        'glue',
        'sst2',
        cache_dir=args.cache_dir,
    )
    dataset = _format_dataset(dataset)

    # Split train/eval, reduce size
    if args.do_eval or args.do_predict:
        eval_dataset = dataset['validation']

        if args.max_eval_samples is not None and len(eval_dataset) > args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))

        if args.group_by_length:
            eval_dataset = eval_dataset.map(lambda x: {'length': len(x['sentence'])})

    if args.do_train:
        train_dataset = dataset['train']

        if args.max_train_samples is not None and len(train_dataset) > args.max_train_samples:
            train_dataset = train_dataset.select(range(args.max_train_samples))

        if args.group_by_length:
            train_dataset = train_dataset.map(lambda x: {'length': len(x['sentence'])})

    data_collator = DataCollatorForSequenceClassification(
        tokenizer=tokenizer,
        source_max_len=args.source_max_len,
        target_max_len=args.target_max_len,
        predict_with_generate=args.predict_with_generate,
    )

    return dict(
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        predict_dataset=eval_dataset if args.do_predict else None,
        data_collator=data_collator
    )
