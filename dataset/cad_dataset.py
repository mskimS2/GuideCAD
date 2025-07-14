import os
import sys
import h5py
import json
import torch
import random
import pickle
from transformers import GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader
from cadlib.macro import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


class CADDataset(Dataset):
    def __init__(self, phase, config, tokenizer=None, prefix_mode: str = "clip"):
        super(CADDataset, self).__init__()
        self.cadvec_path = os.path.join(config.data.cadvec_root)  # h5 data root
        self.prefix_path = os.path.join(config.data.prefix_root)  # image prefix data root
        self.token_path = os.path.join(config.data.gpt_token_root)
        self.phase = phase
        self.path = os.path.join(f"dataset/data/guidecad.json")

        with open(self.path, "r") as fp:
            self.all_data = json.load(fp)[phase]

        self.tokenizer = setup_tokenizer() if not tokenizer else tokenizer
        self.max_length = config.model.max_length
        self.prefix_length = config.model.prefix_length
        self.max_n_loops = config.cad_params.max_n_loops  # Number of paths (N_P)
        self.max_n_curves = config.cad_params.max_n_curves  # Number of commands (N_C)
        self.max_total_len = config.cad_params.max_total_len
        self.prefix_mode = prefix_mode

    def __getitem__(self, index: int):
        data_id = self.all_data[index]

        h5_path = os.path.join(self.cadvec_path, data_id + ".h5")
        with h5py.File(h5_path, "r") as fp:
            cad_vec = fp["vec"][:]  # (len, 1 + N_ARGS)

        pad_len = self.max_total_len - cad_vec.shape[0]
        cad_vec = np.concatenate([cad_vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)

        cmd, param = cad_vec[:, 0], cad_vec[:, 1:]

        prefix = self.get_image_embedding(data_id)

        # FIXME: It would be better to implement this as a collect_fn function.
        tokenized = self.read_preprocessed_token(self.token_path, data_id.split("/")[-1])
        input_ids = tokenized["input_ids"]  # [batch_size, seq_len] -> [seq_len]
        mask = [tokenized["attention_mask"]]  # [batch_size, seq_len]
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long)

        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.long)

        if self.prefix_length > 0:
            mask = self.add_prefix_to_attention_mask(mask, self.prefix_length)  # [batch_size, prefix_length+seq_len]

        mask = mask[0]

        return {
            "command": torch.tensor(cmd, dtype=torch.long),
            "args": torch.tensor(param, dtype=torch.long),
            "id": data_id,
            "prefix": prefix,
            "input_ids": input_ids,
            "mask": mask,
        }

    def __len__(self):
        return len(self.all_data)

    def read_preprocessed_token(self, path: str, data_id: str):
        with open(path + f"/{data_id}.pkl", "rb") as f:
            loaded_data = pickle.load(f)
        return loaded_data

    def add_prefix_to_attention_mask(self, attention_mask: torch.Tensor, prefix_length: int) -> torch.Tensor:
        """
        Add prefix length to attention mask.

        Args:
            attention_mask: Tensor of shape [batch_size, seq_len]
            prefix_length: Length of the prefix to prepend

        Returns:
            Modified attention mask with prefix added: [batch_size, prefix_length + seq_len]
        """
        batch_size = attention_mask.size(0)

        # Create prefix mask (all 1s for prefix length)
        prefix_mask = torch.ones(batch_size, prefix_length, dtype=attention_mask.dtype, device=attention_mask.device)

        # Concatenate prefix mask with the original attention mask
        return torch.cat([prefix_mask, attention_mask], dim=1)  # [batch_size, prefix_length + seq_len]

    def get_image_embedding(self, data_id: int):
        rand_idx = random.randint(0, 6)  # [0,6]
        if self.phase in ["validation", "test"]:
            rand_idx = 6  # (30,30,30 degrees)

        with open(os.path.join(self.prefix_path, data_id + ".pkl"), "rb") as fp:
            prefixes = pickle.load(fp)
            prefix = prefixes["clip_embedding"][rand_idx]

        return prefix

    def get_data_by_id(self, data_id):
        idx = self.all_data.index(data_id)
        return self.__getitem__(idx)


def worker_init_fn(worker_id):
    np.random.seed(torch.initial_seed() % 2**32)


def get_dataloader(phase, config, tokenizer, rank, prefix_mode: str = "clip"):
    world_size = torch.distributed.get_world_size()
    num_workers = config.model.num_workers
    dataset = CADDataset(phase, config, tokenizer, prefix_mode)

    # multi-gpu
    if world_size > 1:
        if phase == "train":
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=torch.distributed.get_world_size(), rank=rank, shuffle=True
            )
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=torch.distributed.get_world_size(), rank=rank, shuffle=False
            )

        local_batch_size = config.model.batch_size // (world_size * config.model.accumulation_steps)

        return DataLoader(
            dataset,
            batch_size=local_batch_size,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
            pin_memory=True if torch.cuda.is_available() else False,
            prefetch_factor=2,
            persistent_workers=True if num_workers > 0 else False,
            sampler=sampler,
        )

    # single-gpu
    local_batch_size = config.model.batch_size // config.model.accumulation_steps
    return DataLoader(
        dataset,
        batch_size=local_batch_size,
        shuffle=phase == "train",
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=True if torch.cuda.is_available() else False,
        prefetch_factor=2,
        persistent_workers=True if num_workers > 0 else False,
    )


def setup_tokenizer(model_name: str = "gpt2") -> GPT2Tokenizer:
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens(
        {
            "pad_token": "[PAD]",
            "bos_token": "<SOS>",
            "eos_token": "<EOS>",
        }
    )

    return tokenizer
