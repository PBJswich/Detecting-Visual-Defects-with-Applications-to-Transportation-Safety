import torch
from torch.utils.data import Dataset
import random

class MultiTaskBatchSampler:
    def __init__(self, datasets, batch_size):
        assert len(datasets) == 3, "Provide three datasets: segmentation, detection, classification."
        assert batch_size % 3 == 0, "Batch size must be divisible by 3."
        self.datasets = datasets
        self.batch_size = batch_size
        self.batch_size_per_task = batch_size // 3
        self.lengths = [len(ds) for ds in datasets]
        self.indices = [list(range(l)) for l in self.lengths]
        self.num_batches = min(l // self.batch_size_per_task for l in self.lengths)

    def __iter__(self):
        # Shuffle indices for each dataset
        self.indices = [random.sample(idxs, len(idxs)) for idxs in self.indices]
        for batch_idx in range(self.num_batches):
            batch = []
            for task_id in range(3):
                start = batch_idx * self.batch_size_per_task
                end = start + self.batch_size_per_task
                batch.extend([(task_id, i) for i in self.indices[task_id][start:end]])
            random.shuffle(batch)  # Shuffle within batch for mixing
            yield batch

    def __len__(self):
        return self.num_batches

class MultiTaskDataLoader:
    def __init__(self, seg_dataset, det_dataset, cls_dataset, batch_size):
        self.datasets = [seg_dataset, det_dataset, cls_dataset]
        self.sampler = MultiTaskBatchSampler(self.datasets, batch_size)
        self.batch_size = batch_size

    def __iter__(self):
        for batch_indices in self.sampler:
            batch = []
            for task_id, idx in batch_indices:
                sample = self.datasets[task_id][idx]
                batch.append(sample)
            yield batch

    def __len__(self):
        return len(self.sampler)

# Example usage:
# from SUTCrackDataset import SUTCrackDataset
# seg_ds = SUTCrackDataset(root_dir, task='segmentation')
# det_ds = SUTCrackDataset(root_dir, task='detection')
# cls_ds = SUTCrackDataset(root_dir, task='classification')
# loader = MultiTaskDataLoader(seg_ds, det_ds, cls_ds, batch_size=12)
# for batch in loader:
#     # batch is a list of dicts, each with 'task_id' and relevant data
#     pass 