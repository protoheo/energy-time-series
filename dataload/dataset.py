from torch.utils.data import Dataset
import torch
import cv2
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, dataframe_set, input_cols, target_cols, mode='train'):
        self.input_values = dataframe_set[input_cols].values
        self.target_values = dataframe_set[target_cols].values
        self.mode = mode

    def __len__(self):
        return len(self.target_values)

    def __getitem__(self, idx):
        input_value = self.input_values[idx]
        target_value = self.target_values[idx]

        input_value, labels = self._data_scale(input_value, target_value)

        return input_value, target_value

    def _data_scale(self, input_path, label_path):
        label_pairs = list(range(1, len(self.label_pair) + 1))

        # Segmentation Gen

        input_value = cv2.imread(input_path)
        labels = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        new_masks = []

        for idx, label in enumerate(label_pairs):
            tmp = np.where(labels == idx, 1, 0)
            new_masks.append(tmp)

        new_masks = np.array(new_masks)
        new_masks = np.moveaxis(np.array(new_masks), 0, -1)

        if self.transform is not None:
            input_value = self.transform(image=input_value, mask=new_masks)
            input_value['mask'] = torch.moveaxis(input_value['mask'], 2, 0)

        return input_value['image'], input_value['mask']


