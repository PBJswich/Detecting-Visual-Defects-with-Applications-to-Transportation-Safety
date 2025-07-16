import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import xml.etree.ElementTree as ET
import numpy as np

class SUTCrackDataset(Dataset):
    def __init__(self, root_dir, task, transform=None, crop_size=1024):
        self.root_dir = root_dir
        self.task = task  # 'segmentation', 'detection', or 'classification'
        self.crop_size = crop_size
        self.transform = transform
        self.positives = self._load_positives()
        self.negatives = self._load_negatives() if task == 'classification' else []
        self.samples = self._build_samples()
        # self.augmentation = transforms.Compose([
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomVerticalFlip(),
        #     transforms.ColorJitter(),
        # ])

    def _load_positives(self):
        img_dir = os.path.join(self.root_dir, '1-Segmentation', 'Original Image')
        mask_dir = os.path.join(self.root_dir, '1-Segmentation', 'Ground Truth')
        bbox_dir = os.path.join(self.root_dir, '2-Object Detection')
        images = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        positives = []
        for img_name in images:
            base = os.path.splitext(img_name)[0]
            mask_name = f"{base}.png"
            bbox_name = f"{base}.xml"
            img_path = os.path.join(img_dir, img_name)
            mask_path = os.path.join(mask_dir, mask_name)
            bbox_path = os.path.join(bbox_dir, bbox_name)
            positives.append({
                'img': img_path,
                'mask': mask_path,
                'bbox': bbox_path,
                'filename': img_name
            })
        return positives

    def _load_negatives(self):
        neg_dir = os.path.join(self.root_dir, '3-Classification', 'Without Crack')
        images = sorted([f for f in os.listdir(neg_dir) if f.endswith('.jpg')])
        negatives = []
        for img_name in images:
            img_path = os.path.join(neg_dir, img_name)
            negatives.append({
                'img': img_path,
                'filename': img_name
            })
        return negatives

    def _build_samples(self):
        samples = []
        if self.task in ['segmentation', 'detection', 'classification']:
            for pos in self.positives:
                samples.append({'type': 'positive', **pos})
        if self.task == 'classification':
            for neg in self.negatives:
                samples.append({'type': 'negative', **neg})
        return samples

    def __len__(self):
        return len(self.samples)

    def _center_crop(self, img):
        w, h = img.size
        crop = self.crop_size
        left = (w - crop) // 2
        top = (h - crop) // 2
        right = left + crop
        bottom = top + crop
        return img.crop((left, top, right, bottom))

    def _parse_xml(self, xml_path):
        bboxes = []
        if not os.path.exists(xml_path):
            return bboxes
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            xmin = int(float(bbox.find('xmin').text))
            ymin = int(float(bbox.find('ymin').text))
            xmax = int(float(bbox.find('xmax').text))
            ymax = int(float(bbox.find('ymax').text))
            bboxes.append([xmin, ymin, xmax, ymax])
        return bboxes

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample['img']).convert('RGB')
        img = self._center_crop(img)
        mask = None
        bboxes = None
        label = None
        if sample['type'] == 'positive':
            if self.task == 'segmentation':
                mask = Image.open(sample['mask']).convert('L')
                mask = self._center_crop(mask)
                mask = torch.from_numpy(np.array(mask)).long()
                label = 1
            elif self.task == 'detection':
                bboxes = self._parse_xml(sample['bbox'])
                label = 1
            elif self.task == 'classification':
                label = 1
        else:  # negative
            if self.task == 'classification':
                label = 0
        # To tensor and normalize
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        return {
            'image': img,
            'mask': mask,
            'bboxes': bboxes,
            'class': label,
            'task_id': self.task,
            'filename': sample['filename']
        } 