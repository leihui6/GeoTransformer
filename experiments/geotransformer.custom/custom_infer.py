import os
import os.path as osp
import time
from xml.parsers.expat import model

import numpy as np
import torch
import torch.utils.data
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from config import make_cfg
from model import create_model
from geotransformer.utils.data import (
    registration_collate_fn_stack_mode,
    calibrate_neighbors_stack_mode,
)
from geotransformer.utils.torch import to_cuda
from geotransformer.modules.registration.metrics import isotropic_transform_error
from geotransformer.utils.pointcloud import apply_transform, get_transform_from_rotation_translation, inverse_transform
from geotransformer.utils.data import precompute_data_stack_mode

# ----------------------------------------------------------------------------- #
# Hard-coded configuration                                                      #
# ----------------------------------------------------------------------------- #
WORKING_DIR = osp.dirname(osp.realpath(__file__))
ROOT_DIR = osp.dirname(osp.dirname(WORKING_DIR))

SOURCE_POINT_CLOUD = r'E:\workspace\PCAlignmentDataGen\src.txt'
# SOURCE_POINT_CLOUD = r'E:\workspace\PCAlignmentDataGen\cloud_bin_0.txt'
TARGET_POINT_CLOUD = r'E:\workspace\PCAlignmentDataGen\tgt.txt'
# TARGET_POINT_CLOUD = r'E:\workspace\PCAlignmentDataGen\cloud_bin_1.txt'

USE_GPU = True  # True：若可用则使用 GPU；False：始终使用 CPU
# SNAPSHOT_PATH = osp.join('./weights', 'geotransformer-3dmatch.pth.tar')
SNAPSHOT_PATH = osp.join('./weights', 'epoch-8.pth.tar')

NEIGHBORSFILE = osp.join(WORKING_DIR, 'neighbor_limits.txt')
# ----------------------------------------------------------------------------- #

def downsample_points(points, num_samples):
    if points.shape[0] <= num_samples:
        return points
    indices = np.random.choice(points.shape[0], num_samples, replace=False)
    return points[indices]

class SinglePairDataset(torch.utils.data.Dataset):
    def __init__(self, ref_points, src_points):
        self.sample = {
            'seq_id': np.int64(0),
            'ref_frame': np.int64(0),
            'src_frame': np.int64(1),
            'ref_points': ref_points.astype(np.float32),
            'src_points': src_points.astype(np.float32),
            'ref_feats': np.ones((ref_points.shape[0], 1), dtype=np.float32),
            'src_feats': np.ones((src_points.shape[0], 1), dtype=np.float32),
            'transform': None,
        }

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.sample


def load_point_cloud(path):
    ext = osp.splitext(path)[1].lower()
    if ext == '.npy':
        points = np.load(path)
    elif ext == '.txt':
        points = np.loadtxt(path)[:, :3]
        points  = downsample_points(points, 6000) 
    elif ext in ['.pcd', '.ply', '.xyz']:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(path)
        if len(pcd.points) == 0:
            raise ValueError(f'Point cloud {path} is empty.')
        points = np.asarray(pcd.points)
    elif ext == '.bin':
        points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]
    else:
        raise ValueError(f'Unsupported point cloud format: {ext}')
    return points.astype(np.float32)


def pose_to_transform(pose, radians=False):
    if pose is None:
        raise ValueError('INITIAL_POSE must be specified when SOURCE_POINT_CLOUD is None.')
    translation = np.asarray(pose[:3], dtype=np.float64)
    angles = np.asarray(pose[3:], dtype=np.float64)
    rotation = Rotation.from_euler('xyz', angles, degrees=not radians).as_matrix()
    transform = get_transform_from_rotation_translation(rotation, translation)
    return transform.astype(np.float32)

def prepare_batch(ref_points, src_points, cfg, neighbor_limits):
    dataset = SinglePairDataset(ref_points, src_points)
    collated = registration_collate_fn_stack_mode(
        [dataset[0]],
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        precompute_data=True,
    )
    return collated

def make_onnx(onnx_path, model, batch):
    import torch.onnx
    torch.onnx.export(
        model,              # model being run
        (batch,),           # model input (or a tuple for multiple inputs)
        onnx_path,          # where to save the model (can be a file or file-like object)
        verbose=True,      # whether to print out a human-readable representation of the model
        input_names = ['input'],   # the model's input names
        output_names = ['output'], # the model's output names
    )

def build_infer_data_dict(ref_points, src_points, cfg, neighbor_limits, device=None):
    """
    ref_points/src_points: np.ndarray or torch.Tensor, (N,3) float32
    returns: data_dict for KPConvFPN/GeoTransformer
    NOTE: geometric precompute runs on CPU only, then moved to device.
    """
    if isinstance(ref_points, np.ndarray):
        ref_points = torch.from_numpy(ref_points)
    if isinstance(src_points, np.ndarray):
        src_points = torch.from_numpy(src_points)

    # default device for network compute
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---------------------------
    # 1) CPU precompute inputs
    # ---------------------------
    ref_cpu = ref_points.detach().cpu().contiguous()
    src_cpu = src_points.detach().cpu().contiguous()

    points0 = torch.cat([ref_cpu, src_cpu], dim=0).contiguous()
    lengths0 = torch.tensor(
        [ref_cpu.shape[0], src_cpu.shape[0]],
        dtype=torch.long,
        device='cpu',
    )

    feats0 = torch.ones((points0.shape[0], 1), dtype=torch.float32, device='cpu')

    # ---------------------------
    # 2) CPU geometric pyramid
    # ---------------------------
    geo = precompute_data_stack_mode(
        points=points0,
        lengths=lengths0,
        num_stages=cfg.backbone.num_stages,
        voxel_size=cfg.backbone.init_voxel_size,
        radius=cfg.backbone.init_radius,
        neighbor_limits=neighbor_limits,
    )

    # ---------------------------
    # 3) move to device
    # ---------------------------
    def move(x):
        if isinstance(x, list):
            return [t.to(device).contiguous() for t in x]
        return x.to(device).contiguous()

    data_dict = {
        "features": feats0.to(device).contiguous(),
        "points": move(geo["points"]),
        "lengths": move(geo["lengths"]),
        "neighbors": move(geo["neighbors"]),
        "subsampling": move(geo["subsampling"]),
        "upsampling": move(geo["upsampling"]),
        "batch_size": 1,
    }
    return data_dict

def main():
    cfg = make_cfg()
    # Load model and weights.
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        if USE_GPU and not torch.cuda.is_available():
            print('Warning: CUDA requested but not available. Falling back to CPU.')
        device = torch.device('cpu')

    model = create_model(cfg).to(device)
    state_dict = torch.load(SNAPSHOT_PATH, map_location='cpu', weights_only=True)
    if 'model' not in state_dict:
        raise RuntimeError('Snapshot does not contain a "model" key.')
    model.load_state_dict(state_dict['model'], strict=True)
    # torch.set_grad_enabled(False)
    model.eval()
    print(f'Model loaded from {SNAPSHOT_PATH}.')

    # Load target point cloud.
    ref_points = load_point_cloud(TARGET_POINT_CLOUD)
    src_points = load_point_cloud(SOURCE_POINT_CLOUD)

    if NEIGHBORSFILE and osp.exists(NEIGHBORSFILE):
        neighbor_limits = np.loadtxt(NEIGHBORSFILE, dtype=np.int64)
        print(f'Neighbor limits loaded from {NEIGHBORSFILE}: {neighbor_limits}')
    else:
        dataset = SinglePairDataset(ref_points, src_points)
        print(f"dataset prepared with {len(dataset)} sample pair(s).")
        neighbor_limits = calibrate_neighbors_stack_mode(
            dataset,
            registration_collate_fn_stack_mode,
            cfg.backbone.num_stages,
            cfg.backbone.init_voxel_size,
            cfg.backbone.init_radius
        )
        np.savetxt(NEIGHBORSFILE, neighbor_limits, fmt='%d')
        print(f'Neighbor limits saved to {NEIGHBORSFILE}.')
    print(f'Neighbor limits calibrated: {neighbor_limits}')
    
    batch = prepare_batch(ref_points, src_points, cfg, neighbor_limits)
    if device.type == 'cuda':
        batch = to_cuda(batch)

    data_dict = build_infer_data_dict(ref_points, src_points, cfg, neighbor_limits, device)
    start_time = time.perf_counter()
    model.eval()
    with torch.no_grad():
        # T_est = model(batch)
        T_est = model.forward_infer(data_dict)
    print (f"est_transform shape: {T_est}")
    end_time = time.perf_counter()

    est_transform_np = T_est.squeeze(0).detach().cpu().numpy()
    
    import open3d as o3d
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    src_t_pcd = o3d.geometry.PointCloud()
    src_t_pcd.points = o3d.utility.Vector3dVector(src_points)
    src_t_pcd.paint_uniform_color([1, 0, 0])
    src_t_pcd.transform(est_transform_np)
    src_t_pcd.paint_uniform_color([0, 0, 1])
    ref_pcd = o3d.geometry.PointCloud()
    ref_pcd.points = o3d.utility.Vector3dVector(ref_points)
    ref_pcd.paint_uniform_color([0, 1, 0]) # green is ref_points
    vis.add_geometry(src_t_pcd)
    vis.add_geometry(ref_pcd)

    # src_pcd = o3d.geometry.PointCloud()
    # src_pcd.points = o3d.utility.Vector3dVector(src_points)
    # src_pcd.paint_uniform_color([1, 0, 0]) # red is src_points
    # vis.add_geometry(src_pcd)
    vis.run()
    vis.destroy_window()
    
    
    make_onnx('geotransformer_custom.onnx', model, (batch,))

if __name__ == '__main__':
    main()
