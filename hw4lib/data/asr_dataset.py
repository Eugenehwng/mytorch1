from typing import Literal, Tuple, Optional
import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset 
from torch.nn.utils.rnn import pad_sequence
try:
    import torchaudio.transforms as tat
except (ImportError, OSError):
    tat = None
from .tokenizer import H4Tokenizer

'''
TODO: Implement this class.

Specification:
The ASRDataset class provides data loading and processing for ASR (Automatic Speech Recognition):

1. Data Organization:
   - Handles dataset partitions (train-clean-100, dev-clean, test-clean)
   - Features stored as .npy files in fbank directory
   - Transcripts stored as .npy files in text directory
   - Maintains alignment between features and transcripts

2. Feature Processing:
   - Loads log mel filterbank features from .npy files
   - Supports multiple normalization strategies:
     * global_mvn: Global mean and variance normalization
     * cepstral: Per-utterance mean and variance normalization
     * none: No normalization
   - Applies SpecAugment data augmentation during training:
     * Time masking: Masks random time steps
     * Frequency masking: Masks random frequency bands

3. Transcript Processing:
   - Similar to LMDataset transcript handling
   - Creates shifted (SOS-prefixed) and golden (EOS-suffixed) versions
   - Tracks statistics for perplexity calculation
   - Handles tokenization using H4Tokenizer

4. Batch Preparation:
   - Pads features and transcripts to batch-uniform lengths
   - Provides lengths for packed sequence processing
   - Ensures proper device placement and tensor types

Key Requirements:
- Must maintain feature-transcript alignment
- Must handle variable-length sequences
- Must track maximum lengths for both features and text
- Must implement proper padding for batching
- Must apply SpecAugment only during training
- Must support different normalization strategies
'''

class ASRDataset(Dataset):
    def __init__(
            self,
            partition:Literal['train-clean-100', 'dev-clean', 'test-clean'],
            config:dict,
            tokenizer:H4Tokenizer,
            isTrainPartition:bool,
            global_stats:Optional[Tuple[torch.Tensor, torch.Tensor]]=None
    ):
        """
        Initialize the ASRDataset for ASR training/validation/testing.
        Args:
            partition (str): Dataset partition ('train-clean-100', 'dev-clean', or 'test-clean')
            config (dict): Configuration dictionary containing dataset settings
            tokenizer (H4Tokenizer): Tokenizer for encoding/decoding text
            isTrainPartition (bool): Whether this is the training partition
                                     Used to determine if SpecAugment should be applied.
            global_stats (tuple, optional): (mean, std) computed from training set.
                                          If None and using global_mvn, will compute during loading.
                                          Should only be None for training set.
                                          Should be provided for dev and test sets.
        """
        self.config = config
        self.partition = partition
        self.isTrainPartition = isTrainPartition
        self.tokenizer = tokenizer

        self.eos_token = tokenizer.eos_id
        self.sos_token = tokenizer.sos_id
        self.pad_token = tokenizer.pad_id

        root = config["root"]
        self.fbank_dir = os.path.join(root, partition, "fbank")
        self.fbank_files = sorted(
            f for f in os.listdir(self.fbank_dir) if f.endswith(".npy")
        )
        subset = float(config.get("subset", 1.0))
        n_keep = max(1, int(len(self.fbank_files) * subset))
        if subset >= 1.0:
            n_keep = len(self.fbank_files)
        self.fbank_files = self.fbank_files[:n_keep]
        self.length = len(self.fbank_files)

        self.text_dir = None
        self.text_files = []
        if self.partition != "test-clean":
            self.text_dir = os.path.join(root, partition, "text")
            self.text_files = sorted(
                f for f in os.listdir(self.text_dir) if f.endswith((".npy", ".txt"))
            )[:n_keep]
            if len(self.fbank_files) != len(self.text_files):
                raise ValueError("Number of feature and transcript files must match")

        self.feats, self.transcripts_shifted, self.transcripts_golden = [], [], []
        self.total_chars = 0
        self.total_tokens = 0
        self.feat_max_len = 0
        self.text_max_len = 0

        if self.config["norm"] == "global_mvn" and global_stats is None:
            if not isTrainPartition:
                raise ValueError(
                    "global_stats must be provided for non-training partitions when using global_mvn"
                )
            count = 0
            mean = torch.zeros(self.config["num_feats"], dtype=torch.float64)
            M2 = torch.zeros(self.config["num_feats"], dtype=torch.float64)

        print(f"Loading data for {partition} partition...")
        for i in tqdm(range(self.length)):
            feat_path = os.path.join(self.fbank_dir, self.fbank_files[i])
            feat = np.load(feat_path)
            nf = self.config["num_feats"]
            feat = feat[:nf, :].astype(np.float32)
            self.feats.append(feat)
            self.feat_max_len = max(self.feat_max_len, feat.shape[1])

            if self.config["norm"] == "global_mvn" and global_stats is None:
                feat_tensor = torch.from_numpy(feat)
                batch_count = feat_tensor.shape[1]
                count += batch_count
                delta = feat_tensor - mean.unsqueeze(1)
                mean += delta.mean(dim=1)
                delta2 = feat_tensor - mean.unsqueeze(1)
                M2 += (delta * delta2).sum(dim=1)

            if self.partition != "test-clean":
                tpath = os.path.join(self.text_dir, self.text_files[i])
                if self.text_files[i].endswith(".npy"):
                    arr = np.load(tpath, allow_pickle=True)
                    transcript = arr.item() if arr.ndim == 0 else str(arr)
                else:
                    with open(tpath, encoding="utf-8") as fh:
                        transcript = fh.read()
                transcript = str(transcript).strip()
                self.total_chars += len(transcript)
                tokenized = self.tokenizer.encode(transcript)
                self.total_tokens += len(tokenized)
                self.text_max_len = max(self.text_max_len, len(tokenized) + 1)
                self.transcripts_shifted.append([self.sos_token] + tokenized)
                self.transcripts_golden.append(tokenized + [self.eos_token])

        self.avg_chars_per_token = (
            self.total_chars / self.total_tokens if self.total_tokens > 0 else 0
        )

        if self.partition != "test-clean":
            if not (
                len(self.feats)
                == len(self.transcripts_shifted)
                == len(self.transcripts_golden)
            ):
                raise ValueError("Features and transcripts are misaligned")

        if self.config["norm"] == "global_mvn":
            if global_stats is not None:
                self.global_mean, self.global_std = global_stats
            else:
                variance = M2 / (count - 1)
                self.global_std = torch.sqrt(variance + 1e-8).float()
                self.global_mean = mean.float()
        else:
            self.global_mean, self.global_std = None, None

        # Initialize SpecAugment transforms (optional if torchaudio is unavailable)
        if tat is not None:
            self.time_mask = tat.TimeMasking(
                time_mask_param=config['specaug_conf']['time_mask_width_range'],
                iid_masks=True
            )
            self.freq_mask = tat.FrequencyMasking(
                freq_mask_param=config['specaug_conf']['freq_mask_width_range'],
                iid_masks=True
            )
        else:
            self.time_mask = None
            self.freq_mask = None

    def get_avg_chars_per_token(self):
        '''
        Get the average number of characters per token. Used to calculate character-level perplexity.
        DO NOT MODIFY
        '''
        return self.avg_chars_per_token

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        DO NOT MODIFY
        """
        return self.length

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.

        Args:
            idx (int): Sample index

        Returns:
            tuple: (features, shifted_transcript, golden_transcript) where:
                - features: FloatTensor of shape (num_feats, time)
                - shifted_transcript: LongTensor (time) or None
                - golden_transcript: LongTensor  (time) or None
        """
        feat = torch.FloatTensor(self.feats[idx].copy())
        if self.config["norm"] == "global_mvn":
            feat = (feat - self.global_mean.unsqueeze(1)) / (
                self.global_std.unsqueeze(1) + 1e-8
            )
        elif self.config["norm"] == "cepstral":
            feat = (feat - feat.mean(dim=1, keepdim=True)) / (
                feat.std(dim=1, keepdim=True) + 1e-8
            )

        shifted_transcript, golden_transcript = None, None
        if self.partition != "test-clean":
            shifted_transcript = torch.LongTensor(self.transcripts_shifted[idx])
            golden_transcript = torch.LongTensor(self.transcripts_golden[idx])

        return feat, shifted_transcript, golden_transcript

    def collate_fn(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Collate and pad a batch of samples to create a batch of fixed-length padded features and transcripts.

        Args:
            batch (list): List of samples from __getitem__

        Returns:
            tuple: (padded_features, padded_shifted, padded_golden, feat_lengths, transcript_lengths) where:
                - padded_features: Tensor of shape (batch, max_time, num_feats)
                - padded_shifted: Tensor of shape (batch, max_len) or None
                - padded_golden: Tensor of shape (batch, max_len) or None  
                - feat_lengths: Tensor of original feature lengths of shape (batch)
                - transcript_lengths: Tensor of transcript lengths of shape (batch) or None
        """
        feats = [b[0].transpose(0, 1) for b in batch]
        feat_lengths = torch.tensor([f.size(0) for f in feats], dtype=torch.long)
        padded_feats = pad_sequence(feats, batch_first=True, padding_value=0.0)

        padded_shifted, padded_golden, transcript_lengths = None, None, None
        if self.partition != "test-clean":
            batch_shifted = [b[1] for b in batch]
            batch_golden = [b[2] for b in batch]
            transcript_lengths = torch.tensor(
                [len(s) for s in batch_shifted], dtype=torch.long
            )
            padded_shifted = pad_sequence(
                batch_shifted, batch_first=True, padding_value=self.pad_token
            )
            padded_golden = pad_sequence(
                batch_golden, batch_first=True, padding_value=self.pad_token
            )

        if self.config["specaug"] and self.isTrainPartition and self.time_mask is not None:
            x = padded_feats.transpose(1, 2)
            if self.config["specaug_conf"]["apply_freq_mask"]:
                for _ in range(self.config["specaug_conf"]["num_freq_mask"]):
                    x = self.freq_mask(x)
            if self.config["specaug_conf"]["apply_time_mask"]:
                for _ in range(self.config["specaug_conf"]["num_time_mask"]):
                    x = self.time_mask(x)
            padded_feats = x.transpose(1, 2)

        return (
            padded_feats,
            padded_shifted,
            padded_golden,
            feat_lengths,
            transcript_lengths,
        )

