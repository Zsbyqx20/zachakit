import os
import random
import shutil
import time
from itertools import chain
from typing import Literal

from datasets import DatasetDict, load_dataset, load_from_disk
from datasets.utils.logging import disable_progress_bar
from pydantic import BaseModel, model_validator
from transformers import AutoTokenizer


class UnitDS(BaseModel):
    cache_dir: str = None
    # whether the dataset is a huggingface one or a raw-text one;
    from_local_file: bool = False
    text_col: str = None
    root: str = None
    data_type: Literal["text", "arrow", "csv", "parquet"] = "arrow"
    # tokenize & group params
    block_size: int = None
    tokenizer_name: str = None

    _dataset = None
    _alive = None

    @model_validator(mode="after")
    def check_root_existence(self):
        if not os.path.exists(self.root):
            raise ValueError(f"The root file/dir {self.root} does not exist!")
        return self

    @model_validator(mode="after")
    def check_tokenizer_validity(self):
        _ = AutoTokenizer.from_pretrained(self.tokenizer_name)
        return self

    def living_check(self):
        try:
            if self.from_local_file:
                # in case local file will be large, use multi-processing
                self._dataset = load_dataset(self.data_type, data_files=self.root, cache_dir=self.cache_dir, num_proc=8)
                if isinstance(self._dataset, DatasetDict):
                    assert "train" in self._dataset
                    self._dataset = self._dataset["train"]
            else:
                self._dataset = load_dataset(self.data_type, data_files=self.root, cache_dir=self.cache_dir)
        except:
            print(f"The dataset loaded from {self.root} cannot be loaded normally.")
            self._alive = False
        else:
            self._alive = True

    def tokenize_group(self):
        assert self.text_col in self._dataset.features, "Wrong column name specified!"

        cache_dir = os.path.join(self.cache_dir, "tokenized")
        save_dir = os.path.join(self.cache_dir, "grouped")
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        # check whether current dataset has already been prepared
        try:
            _ = load_from_disk(save_dir)
        except:
            # need tokenizing
            def group_texts(examples):
                concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
                total_length = len(concatenated_examples[list(examples.keys())[0]])
                total_length = (total_length // self.block_size) * self.block_size
                result = {
                    k: [t[i : i + self.block_size] for i in range(0, total_length, self.block_size)]
                    for k, t in concatenated_examples.items()
                }
                result["labels"] = result["input_ids"].copy()
                return result

            if not os.path.exists(cache_dir):
                os.mkdir(cache_dir)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            time.sleep(random.randint(0, 5))

            # disable progress bar for prettier console
            disable_progress_bar()
            tokenized_ds = self._dataset.map(
                lambda examples: tokenizer(examples[self.text_col]),
                batched=True,
                num_proc=8,
                remove_columns=self._dataset.features,
                load_from_cache_file=True,
                cache_file_name=os.path.join(cache_dir, "tokenized.arrow"),
                desc="Tokenizing dataset",
            )
            grouped_ds = tokenized_ds.map(
                group_texts,
                batched=True,
                num_proc=64,
                load_from_cache_file=True,
                cache_file_name=os.path.join(cache_dir, "grouped.arrow"),
                desc="Grouping texts",
            )

            grouped_ds.save_to_disk(save_dir)

        # clean tokenizing directory to free space.
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        return True
