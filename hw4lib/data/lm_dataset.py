from typing import Tuple, List
import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from .tokenizer import H4Tokenizer


class LMDataset(Dataset):
    """
    Dataset for Language Model training/evaluation.
    """

    def __init__(self, partition: str, config: dict, tokenizer: H4Tokenizer):
        self.config = config
        self.partition = partition
        self.tokenizer = tokenizer

        self.eos_token = tokenizer.eos_id
        self.sos_token = tokenizer.sos_id
        self.pad_token = tokenizer.pad_id

        self.text_dir = os.path.join(config["root"], partition, "text")
        self.text_files = sorted(
            f
            for f in os.listdir(self.text_dir)
            if os.path.isfile(os.path.join(self.text_dir, f))
            and (f.endswith(".npy") or f.endswith(".txt"))
        )
        subset = float(config.get("subset", 1.0))
        n_keep = max(1, int(len(self.text_files) * subset))
        if subset >= 1.0:
            n_keep = len(self.text_files)
        self.text_files = self.text_files[:n_keep]

        self.transcripts_shifted: List[List[int]] = []
        self.transcripts_golden: List[List[int]] = []

        self.total_chars = 0
        self.total_tokens = 0
        self.text_max_len = 0

        print(f"Loading transcripts for {partition} partition...")
        for file in tqdm(self.text_files):
            path = os.path.join(self.text_dir, file)
            if file.endswith(".npy"):
                arr = np.load(path, allow_pickle=True)
                transcript = arr.item() if arr.ndim == 0 else str(arr)
            else:
                with open(path, encoding="utf-8") as fh:
                    transcript = fh.read()
            transcript = str(transcript).strip()
            self.total_chars += len(transcript)
            tokenized = self.tokenizer.encode(transcript)
            self.total_tokens += len(tokenized)
            self.text_max_len = max(self.text_max_len, len(tokenized) + 1)

            shifted = [self.sos_token] + tokenized
            golden = tokenized + [self.eos_token]
            self.transcripts_shifted.append(shifted)
            self.transcripts_golden.append(golden)

        self.avg_chars_per_token = (
            self.total_chars / self.total_tokens if self.total_tokens > 0 else 0
        )

        if not (len(self.transcripts_shifted) == len(self.transcripts_golden)):
            raise ValueError("Shifted and golden transcripts are misaligned")

        self.length = len(self.transcripts_shifted)

    def get_avg_chars_per_token(self) -> float:
        return self.avg_chars_per_token

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.LongTensor, torch.LongTensor]:
        shifted = torch.LongTensor(self.transcripts_shifted[idx])
        golden = torch.LongTensor(self.transcripts_golden[idx])
        return shifted, golden

    def collate_fn(
        self, batch: List[Tuple[torch.LongTensor, torch.LongTensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shifted_transcripts, golden_transcripts = zip(*batch)
        lengths = torch.tensor([len(s) for s in shifted_transcripts], dtype=torch.long)
        padded_shifted = pad_sequence(
            shifted_transcripts, batch_first=True, padding_value=self.pad_token
        )
        padded_golden = pad_sequence(
            golden_transcripts, batch_first=True, padding_value=self.pad_token
        )
        return padded_shifted, padded_golden, lengths

    def sample_prompts(self, num_samples: int, prompt_length: int, seed: int = None):
        if seed is not None:
            np_state = np.random.get_state()
            np.random.seed(seed)

        prompts = []
        originals = []
        attempts = 0
        max_attempts = num_samples * 10

        while len(prompts) < num_samples and attempts < max_attempts:
            idx = np.random.randint(0, len(self))
            tokens = self.transcripts_shifted[idx][1:]
            if len(tokens) < prompt_length:
                attempts += 1
                continue
            prompt_tokens = tokens[:prompt_length]
            prompts.append(torch.LongTensor([self.sos_token] + prompt_tokens))
            originals.append(torch.LongTensor(tokens + [self.eos_token]))
            attempts += 1

        if len(prompts) < num_samples:
            print(f"Warning: Could only sample {len(prompts)} valid prompts")

        if seed is not None:
            np.random.set_state(np_state)

        if len(prompts) == 0:
            return torch.empty(0, 1, dtype=torch.long), []

        return torch.stack(prompts), originals
