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
        # overlap_threshold=None,
        # return_corr_indices=False,
        # matching_radius=None,
        add_rotate=True,
    ):
        super(WeldSeamDataset, self).__init__()

        self.dataset_root = dataset_root
        # self.metadata_root = osp.join(self.dataset_root, 'metadata')
        # self.data_root = osp.join(self.dataset_root, 'data')

        self.subset = subset
        self.point_limit = point_limit
        # self.overlap_threshold = overlap_threshold
        self.add_rotate = add_rotate

        # self.return_corr_indices = return_corr_indices
        # self.matching_radius = matching_radius
        # if self.return_corr_indices and self.matching_radius is None:
            # raise ValueError('"matching_radius" is None but "return_corr_indices" is set.')

        self.use_augmentation = use_augmentation
        self.aug_noise = augmentation_noise
        self.aug_rotation = augmentation_rotation

        with open(osp.join(self.dataset_root, f'{subset}.pkl'), 'rb') as f:
            self.metadata_list = pickle.load(f)
        #     if self.overlap_threshold is not None:
        #         self.metadata_list = [x for x in self.metadata_list if x['overlap'] > self.overlap_threshold]

    def __len__(self):
        return len(self.metadata_list)

    def _crop_point_cloud(self, points: np.ndarray):
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

        # metadata
        metadata: Dict = self.metadata_list[index]
        # data_dict['scene_name'] = metadata['scene_name']
        # data_dict['ref_frame'] = metadata['src']
        # data_dict['src_frame'] = metadata['frag_id1']
        # data_dict['overlap'] = metadata['overlap']

        # get transformation
        rotation = metadata['rot']
        translation = metadata['trans']

        # get point cloud
        src_points = self._crop_point_cloud(metadata['src'])
        ref_points = self._crop_point_cloud(metadata['ref'])

        # src_points = normalize_points(src_points)
        # ref_points = normalize_points(ref_points)

        # augmentation
        if self.use_augmentation:
            ref_points, src_points, rotation, translation = self._augment_point_cloud(
                ref_points, src_points, rotation, translation
            )

        # if self.add_rotate:
        #     ref_rotation = random_sample_rotation_v2()
        #     ref_points = np.matmul(ref_points, ref_rotation.T)
        #     rotation = np.matmul(ref_rotation, rotation)
        #     translation = np.matmul(ref_rotation, translation)

        #     src_rotation = random_sample_rotation_v2()
        #     src_points = np.matmul(src_points, src_rotation.T)
        #     rotation = np.matmul(rotation, src_rotation.T)

        transform = get_transform_from_rotation_translation(rotation, translation)

        # get correspondences
        # if self.return_corr_indices:
        #     corr_indices = get_correspondences(ref_points, src_points, transform, self.matching_radius)
        #     data_dict['corr_indices'] = corr_indices

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