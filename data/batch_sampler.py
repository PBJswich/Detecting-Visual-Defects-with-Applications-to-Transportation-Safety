class TaskSpecificBatchSampler:
    def __init__(self, dataset, batch_size, task_ratios={'seg': 0.33, 'det': 0.33, 'cls': 0.34}):
        self.dataset = dataset
        self.batch_size = batch_size
        self.task_ratios = task_ratios
        
    def __iter__(self):
        task_indices = {
            'seg': [i for i, sample in enumerate(self.dataset) if sample['has_mask']],
            'det': [i for i, sample in enumerate(self.dataset) if sample['has_bbox']],
            'cls': list(range(len(self.dataset)))
        }
        # Implementation of balanced sampling logic