# /Users/louloute/PycharmProjects/INF8225_projet/dataset.py

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Optional

class ALOHADataset(Dataset):
    def __init__(self, hf_dataset, task_specific: bool = False, task: Optional[str] = None):
        self.hf_dataset = hf_dataset

        if task_specific and task:
            self.indices = [i for i, sample in enumerate(hf_dataset) if sample["task_tag"] == task]
        else:
            self.indices = list(range(len(hf_dataset)))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        sample = self.hf_dataset[self.indices[idx]]

        if "input_features" in sample:
            features = torch.tensor(sample["input_features"], dtype=torch.float32)
        else:
            features = []
            for key, value in sample.items():
                if key not in ["task_tag", "labels"] and isinstance(value, (list, np.ndarray)):
                    tensor_val = torch.tensor(value, dtype=torch.float32)
                    features.append(tensor_val)

            if features:
                features = torch.cat(features, dim=-1)
            else:
                features = torch.tensor([], dtype=torch.float32)

        labels = torch.tensor(sample.get("labels", [0.0]), dtype=torch.float32)

        task_tag = sample.get("task_tag", "transfer")
        task_id = 0 if task_tag == "transfer" else 1
        task_tensor = torch.tensor([task_id], dtype=torch.long)

        return {
            "features": features,
            "labels": labels,
            "task_id": task_tensor
        }

    def __repr__(self):
        return f"<ALOHADataset size={len(self)} | task_specific={len(self.indices) != len(self.hf_dataset)}>"