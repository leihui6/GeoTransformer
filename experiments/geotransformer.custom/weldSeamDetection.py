import os.path as osp
import pickle
import random
from typing import Dict
import os
import numpy as np
import torch
import torch.utils.data

from geotransformer.utils.pointcloud import (
    random_sample_rotation,
    random_sample_rotation_v2,
    get_transform_from_rotation_translation,
)

from geotransformer.transforms.functional import (
    normalize_points,
    # random_jitter_points,
    # random_shuffle_points,
    # random_sample_points,
    # random_crop_point_cloud_with_plane,
    # random_sample_viewpoint,
    # random_crop_point_cloud_with_point,
)

from geotransformer.utils.registration import get_correspondences


class WeldSeamDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_root,
        subset,
        point_limit=None,
        use_augmentation=False,
        augmentation_noise=0.003,
        augmentation_rotation=1,
        add_rotate=True,
    ):
        super(WeldSeamDataset, self).__init__()

        self.dataset_root = dataset_root
        self.subset = subset
        self.point_limit = point_limit
        self.add_rotate = add_rotate
        self.use_augmentation = use_augmentation
        self.aug_noise = augmentation_noise
        self.aug_rotation = augmentation_rotation

        with open(osp.join(self.dataset_root, f'{subset}.pkl'), 'rb') as f:
            self.metadata_list = pickle.load(f)

    def __len__(self):
        return len(self.metadata_list)

    def downsample(self, points: np.ndarray):
        # NOTE: setting "point_limit" with "num_workers" > 1 will cause nondeterminism.
        if self.point_limit is not None and points.shape[0] > self.point_limit:
            indices = np.random.permutation(points.shape[0])[: self.point_limit]
            points = points[indices]
        return points

    def _augment_point_cloud(self, ref_points, src_points, rotation, translation):
        r"""Augment point clouds.
        ref_points = src_points @ rotation.T + translation
        1. Random rotation to one point cloud.
        2. Random noise.
        """
        aug_rotation = random_sample_rotation(self.aug_rotation)
        if random.random() > 0.5:
            ref_points = np.matmul(ref_points, aug_rotation.T)
            rotation = np.matmul(aug_rotation, rotation)
            translation = np.matmul(aug_rotation, translation)
        else:
            src_points = np.matmul(src_points, aug_rotation.T)
            rotation = np.matmul(rotation, aug_rotation.T)

        ref_points += (np.random.rand(ref_points.shape[0], 3) - 0.5) * self.aug_noise
        src_points += (np.random.rand(src_points.shape[0], 3) - 0.5) * self.aug_noise

        return ref_points, src_points, rotation, translation

    def __getitem__(self, index):
        data_dict = {}
        # get transformation
        rotation = self.metadata_list[index]['rot']
        translation = self.metadata_list[index]['trans']

        # get point cloud
        src_points = self.downsample(self.metadata_list[index]['src'][:, :3])
        ref_points = self.downsample(self.metadata_list[index]['ref'][:, :3])

        # augmentation
        if self.use_augmentation:
            ref_points, src_points, rotation, translation = self._augment_point_cloud(
                ref_points, src_points, rotation, translation
            )

        transform = get_transform_from_rotation_translation(rotation, translation)

        data_dict['ref_points'] = ref_points.astype(np.float32)
        data_dict['src_points'] = src_points.astype(np.float32)
        data_dict['ref_feats'] = np.ones((ref_points.shape[0], 1), dtype=np.float32)
        data_dict['src_feats'] = np.ones((src_points.shape[0], 1), dtype=np.float32)
        data_dict['transform'] = transform.astype(np.float32)

        # Debug save
        os.makedirs('debug', exist_ok=True)
        np.savetxt(osp.join('debug', f'ref_{index}.txt'), ref_points, fmt='%.6f')
        np.savetxt(osp.join('debug', f'src_{index}.txt'), src_points, fmt='%.6f')
        np.savetxt(osp.join('debug', f'transform_{index}.txt'), transform, fmt='%.6f')
        exit()
        return data_dict