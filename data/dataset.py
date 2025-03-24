# Copyright (c) Duowang Zhu.
# All rights reserved.

import cv2
from skimage import io
import numpy as np
import torch.utils.data
import os
import h5py
import json
from os.path import join as osp
from typing import List, Tuple, Optional, Dict, Any, Union


class BCDDataset(torch.utils.data.Dataset):
    """
    Dataset class for loading and processing bi-temporal remote sensing images.
    
    This class handles loading image pairs (pre and post) along with their 
    corresponding change detection ground truth masks.
    """

    def __init__(self, file_root: str, split: str, transform = None):
        """
        Initialize the dataset with paths to images and transforms.
        
        Args:
            dataset: Dataset split name (e.g., 'train', 'val', 'test')
            file_root: Root directory containing dataset folders
            transform: Optional transform to apply to images and labels
        """
        # Validate inputs
        if not os.path.exists(file_root):
            raise FileNotFoundError(f"Dataset root path does not exist: {file_root}")
        
        # file list
        self.file_list = os.listdir(osp(file_root, split, 'label'))
            
        # Create paths for pre-change images, post-change images, and labels
        self.pre_images = [osp(file_root, split, 't1', x) for x in self.file_list]
        self.post_images = [osp(file_root, split, 't2', x) for x in self.file_list]
        self.label_change = [osp(file_root, split, 'label', x) for x in self.file_list]

        # Store transform
        self.transform = transform
        
        # Validate that all files exist
        self._validate_files()
        
    def _validate_files(self) -> None:
        """Validate that all image files exist."""
        for idx, (pre, post, lbl_c) in enumerate(zip(self.pre_images, self.post_images, self.label_change)):
            if not os.path.exists(pre):
                raise FileNotFoundError(f"Pre-change image not found: {pre}")
            if not os.path.exists(post):
                raise FileNotFoundError(f"Post-change image not found: {post}")
            if not os.path.exists(lbl_c):
                raise FileNotFoundError(f"Ground truth mask not found: {lbl_c}")

    def __len__(self) -> int:
        """Return the number of image pairs in the dataset."""
        return len(self.label_change)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the image pair and label at the specified index.
        
        Args:
            idx: Index of the sample to fetch
            
        Returns:
            Tuple containing:
                - Concatenated pre/post images [H, W, 6]
                - Binary change detection mask [H, W]
        """
        # Load images
        pre_image = io.imread(self.pre_images[idx])
        post_image = io.imread(self.post_images[idx])
        label = io.imread(self.label_change[idx], as_gray=True)

        # Error handling for image loading failures
        if pre_image is None:
            raise IOError(f"Failed to load pre-change image: {self.pre_images[idx]}")
        if post_image is None:
            raise IOError(f"Failed to load post-change image: {self.post_images[idx]}")
        if label is None:
            raise IOError(f"Failed to load ground truth mask: {self.gts[idx]}")
            
        # Concatenate pre and post images along the channel dimension
        # This creates a 6-channel image (BGR-BGR)
        img = np.concatenate((pre_image, post_image), axis=2)

        # Apply transforms if specified
        if self.transform:
            img, label = self.transform(img, label)

        return img, label

    def get_img_info(self, idx: int) -> Dict[str, int]:
        """
        Get image dimensions for the specified index.
        
        Args:
            idx: Index of the image
            
        Returns:
            Dictionary with image height and width
        """
        img = cv2.imread(self.pre_images[idx])
        if img is None:
            raise IOError(f"Failed to load image for info: {self.pre_images[idx]}")
            
        return {"height": img.shape[0], "width": img.shape[1]}
    

class SCDDataset(torch.utils.data.Dataset):
    """
    Dataset class for loading and processing bi-temporal remote sensing images.
    
    This class handles loading image pairs (pre and post) along with their 
    corresponding semantic masks and change detection ground truth.
    """

    def __init__(self, file_root: str, split: str, transform = None):
        """
        Initialize the dataset with paths to images and transforms.
        
        Args:
            dataset: Dataset split name (e.g., 'train', 'val', 'test')
            file_root: Root directory containing dataset folders
            transform: Optional transform to apply to images and labels
        """
        # Validate inputs
        if not os.path.exists(file_root):
            raise FileNotFoundError(f"Dataset root path does not exist: {file_root}")
        
        # file list
        self.file_list = os.listdir(osp(file_root, split, 'label1'))
            
        # Create paths for pre-change images, post-change images, and labels
        self.pre_images = [osp(file_root, split, 't1', x) for x in self.file_list]
        self.post_images = [osp(file_root, split, 't2', x) for x in self.file_list]
        self.pre_label = [osp(file_root, split, 'label1', x) for x in self.file_list]
        self.post_label = [osp(file_root, split, 'label2', x) for x in self.file_list]
        self.label_change = [osp(file_root, split, 'change', x) for x in self.file_list]

        # Store transform
        self.transform = transform
        
        # Validate that all files exist
        self._validate_files()
        
    def _validate_files(self) -> None:
        """Validate that all image files exist."""
        for idx, (img_pre, img_post, lbl_pre, lbl_post, lbl_c) in enumerate(
                zip(self.pre_images, self.post_images, self.pre_label, self.post_label, self.label_change)):
            if not os.path.exists(img_pre):
                raise FileNotFoundError(f"Pre-change image not found: {img_pre}")
            if not os.path.exists(img_post):
                raise FileNotFoundError(f"Post-change image not found: {img_post}")
            if not os.path.exists(lbl_pre):
                raise FileNotFoundError(f"Pre-change label not found: {lbl_pre}")
            if not os.path.exists(lbl_post):
                raise FileNotFoundError(f"Post-change label not found: {lbl_post}")
            if not os.path.exists(lbl_c):
                raise FileNotFoundError(f"Change mask not found: {lbl_c}")

    def __len__(self) -> int:
        """Return the number of image pairs in the dataset."""
        return len(self.label_change)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the image pair and label at the specified index.
        
        Args:
            idx: Index of the sample to fetch
            
        Returns:
            Tuple containing:
                - Concatenated pre/post images [H, W, 6]
                - Binary change detection mask [H, W]
        """
        # Load images
        pre_image = io.imread(self.pre_images[idx])
        post_image = io.imread(self.post_images[idx])
        pre_label = io.imread(self.pre_label[idx], as_gray=True).astype(np.uint8)
        post_label = io.imread(self.post_label[idx], as_gray=True).astype(np.uint8)
        label_change = io.imread(self.label_change[idx], as_gray=True).astype(np.uint8)
        
        # Error handling for image loading failures
        if pre_image is None:
            raise IOError(f"Failed to load pre-change image: {self.pre_images[idx]}")
        if post_image is None:
            raise IOError(f"Failed to load post-change image: {self.post_images[idx]}")
        if pre_label is None:
            raise IOError(f"Failed to load ground truth mask: {self.pre_label[idx]}")
        if post_label is None:
            raise IOError(f"Failed to load ground truth mask: {self.post_label[idx]}")
        if label_change is None:
            raise IOError(f"Failed to load ground truth mask: {self.label_change[idx]}")
            
        # Concatenate pre and post images along the channel dimension
        img = np.concatenate((pre_image, post_image), axis=2)
        label = np.concatenate((pre_label[..., None], post_label[..., None], label_change[..., None]), axis=2)

        # Apply transforms if specified
        if self.transform:
            img, label = self.transform(img, label)

        return img, label

    def get_img_info(self, idx: int) -> Dict[str, int]:
        """
        Get image dimensions for the specified index.
        
        Args:
            idx: Index of the image
            
        Returns:
            Dictionary with image height and width
        """
        img = cv2.imread(self.pre_images[idx])
        if img is None:
            raise IOError(f"Failed to load image for info: {self.pre_images[idx]}")
            
        return {"height": img.shape[0], "width": img.shape[1]}
    

class BDADataset(torch.utils.data.Dataset):
    """
    Dataset class for loading and processing bi-temporal remote sensing images.
    
    This class handles loading image pairs (pre and post) along with their 
    corresponding change detection ground truth masks.
    """

    def __init__(self, file_root: str, split: str, transform = None):
        """
        Initialize the dataset with paths to images and transforms.
        
        Args:
            dataset: Dataset split name (e.g., 'train', 'val', 'test')
            file_root: Root directory containing dataset folders
            transform: Optional transform to apply to images and labels
        """
        # Validate inputs
        if not os.path.exists(file_root):
            raise FileNotFoundError(f"Dataset root path does not exist: {file_root}")
        
        # file list
        self.file_list = os.listdir(osp(file_root, split, 't1'))
            
        # Create paths for pre-change images, post-change images, and labels
        self.pre_images = [osp(file_root, split, 't1', x) for x in self.file_list]
        self.post_images = [osp(file_root, split, 't2', x) for x in self.file_list]
        self.label_loc = [osp(file_root, split, 'label1', x.replace('disaster', 'disaster_target')) for x in self.file_list]
        self.label_cls = [osp(file_root, split, 'label2', x.replace('disaster', 'disaster_target')) for x in self.file_list]

        # Store transform
        self.transform = transform
        
        # Validate that all files exist
        self._validate_files()
        
    def _validate_files(self) -> None:
        """Validate that all image files exist."""
        for idx, (pre, post, lbl_loc, lbl_cls) in enumerate(zip(self.pre_images, self.post_images, self.label_loc, self.label_cls)):
            if not os.path.exists(pre):
                raise FileNotFoundError(f"Pre-change image not found: {pre}")
            if not os.path.exists(post):
                raise FileNotFoundError(f"Post-change image not found: {post}")
            if not os.path.exists(lbl_loc):
                raise FileNotFoundError(f"Ground truth mask not found: {lbl_loc}")
            if not os.path.exists(lbl_cls):
                raise FileNotFoundError(f"Ground truth mask not found: {lbl_cls}")

    def __len__(self) -> int:
        """Return the number of image pairs in the dataset."""
        return len(self.label_loc)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the image pair and label at the specified index.
        
        Args:
            idx: Index of the sample to fetch
            
        Returns:
            Tuple containing:
                - Concatenated pre/post images [H, W, 6]
                - Binary change detection mask [H, W]
        """
        # Load images
        pre_image = cv2.imread(self.pre_images[idx])
        post_image = cv2.imread(self.post_images[idx])
        label_loc = cv2.imread(self.label_loc[idx], 0).astype(np.uint8)
        label_cls = cv2.imread(self.label_cls[idx], 0).astype(np.uint8)

        # Error handling for image loading failures
        if pre_image is None:
            raise IOError(f"Failed to load pre-change image: {self.pre_images[idx]}")
        if post_image is None:
            raise IOError(f"Failed to load post-change image: {self.post_images[idx]}")
        if label_loc is None:
            raise IOError(f"Failed to load ground truth mask: {self.label_loc[idx]}")
        if label_cls is None:
            raise IOError(f"Failed to load ground truth mask: {self.label_cls[idx]}")
            
        # Concatenate pre and post images along the channel dimension
        # This creates a 6-channel image (BGR-BGR)
        img = np.concatenate((pre_image, post_image), axis=2)
        label = np.concatenate((label_loc[:, :, None], label_cls[:, :, None]), axis=2)

        # Apply transforms if specified
        if self.transform:
            img, label = self.transform(img, label)
        
        return img, label

    def get_img_info(self, idx: int) -> Dict[str, int]:
        """
        Get image dimensions for the specified index.
        
        Args:
            idx: Index of the image
            
        Returns:
            Dictionary with image height and width
        """
        img = cv2.imread(self.pre_images[idx])
        if img is None:
            raise IOError(f"Failed to load image for info: {self.pre_images[idx]}")
            
        return {"height": img.shape[0], "width": img.shape[1]}


class CaptionDataset(torch.utils.data.Dataset):
    """
    Dataset class for loading image-caption pairs for change captioning.
    
    This class handles loading image pairs along with their corresponding 
    captions and caption lengths from HDF5 and JSON files.
    """

    def __init__(self, file_root: str, dataset: str, split: str, transform = None):
        """
        Initialize the dataset with paths to images, captions, and transforms.
        
        Args:
            file_root: Folder where data files are stored
            dataset: Base name of processed datasets
            split: Dataset split, one of 'TRAIN', 'VAL', or 'TEST'
            transform: Optional transform to apply to images
        """
        # Validate inputs
        self.split = split
        if self.split not in {'TRAIN', 'VAL', 'TEST'}:
            raise ValueError(f"Split {split} not recognized. Must be one of 'TRAIN', 'VAL', or 'TEST'")
        
        if not os.path.exists(file_root):
            raise FileNotFoundError(f"Dataset folder path does not exist: {file_root}")
            
        # Define file paths
        images_path = os.path.join(file_root, f'{self.split}_IMAGES_{dataset}.hdf5')
        captions_path = os.path.join(file_root, f'{self.split}_CAPTIONS_{dataset}.json')
        caplens_path = os.path.join(file_root, f'{self.split}_CAPLENS_{dataset}.json')
        
        # Validate file existence
        for path in [images_path, captions_path, caplens_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required data file not found: {path}")
        
        # Open HDF5 file where images are stored
        self.h = h5py.File(images_path, 'r')
        self.imgs = self.h['images']

        # Number of captions per image
        self.cpi = self.h.attrs['captions_per_image']

        # Load encoded captions into memory
        with open(captions_path, 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths into memory
        with open(caplens_path, 'r') as j:
            self.caplens = json.load(j)

        # PyTorch transformation pipeline for the image
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions)

    def __len__(self) -> int:
        """Return the number of caption entries in the dataset."""
        return self.dataset_size

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
                                             Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Get the image pair, caption, and caption length at the specified index.
        
        Args:
            idx: Index of the sample to fetch
            
        Returns:
            For training: Tuple (image, caption, caption_length)
            For validation/testing: Tuple (image, caption, caption_length, all_captions)
        """
        # The idx caption corresponds to the (idx // captions_per_image)th image
        img_idx = idx // self.cpi
        img = torch.FloatTensor(self.imgs[img_idx] / 255.)
        
        # Apply transformations if provided
        if self.transform is not None:
            if img.shape == torch.Size([3, 256, 256]):
                # Single image case
                img = self.transform(img)
            elif img.shape == torch.Size([2, 3, 256, 256]):
                # Image pair case
                img[0] = self.transform(img[0])
                img[1] = self.transform(img[1])

        # Data augmentation: randomly swap image pairs during training
        if self.split == 'TRAIN' and np.random.rand() < 0.3:
            img[0], img[1] = img[1].clone(), img[0].clone()

        # Load caption and its length
        caption = torch.LongTensor(self.captions[idx])
        caplen = torch.LongTensor([self.caplens[idx]])

        # Return appropriate data based on split
        if self.split == 'TRAIN':
            return img, caption, caplen
        else:
            # For validation or testing, also return all captions for this image to compute BLEU-4 score
            start_idx = img_idx * self.cpi
            end_idx = start_idx + self.cpi
            all_captions = torch.LongTensor(self.captions[start_idx:end_idx])
            return img, caption, caplen, all_captions
    
    def get_image_captions(self, img_idx: int) -> List[List[int]]:
        """
        Get all captions for a specific image.
        
        Args:
            img_idx: Index of the image
            
        Returns:
            List of caption token sequences for the specified image
        """
        start_idx = img_idx * self.cpi
        end_idx = start_idx + self.cpi
        return self.captions[start_idx:end_idx]
    
    def close(self) -> None:
        """Close the HDF5 file."""
        if hasattr(self, 'h'):
            self.h.close()