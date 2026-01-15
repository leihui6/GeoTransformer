import numpy as np
import torch

from geotransformer.utils.data import (
    registration_collate_fn_stack_mode,
    calibrate_neighbors_stack_mode,
)
from geotransformer.utils.torch import to_cuda
from geotransformer.utils.pointcloud import apply_transform
from model import create_model
from config import make_cfg


class _SinglePairDataset:
    """极简 dataset，只为 calibrate_neighbors / collate 使用"""

    def __init__(self, ref_points, src_points, transform):
        self.sample = {
            'seq_id': np.int64(0),
            'ref_frame': np.int64(0),
            'src_frame': np.int64(1),
            'ref_points': ref_points.astype(np.float32),
            'src_points': src_points.astype(np.float32),
            'ref_feats': np.ones((ref_points.shape[0], 1), dtype=np.float32),
            'src_feats': np.ones((src_points.shape[0], 1), dtype=np.float32),
            'transform': transform.astype(np.float32),
        }

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.sample


class GeoTransformerInfer:
    """
    GeoTransformer inference engine
    输入:  src_points (N,3), ref_points (M,3)
    输出:  4x4 matrix (src -> ref)
    """

    def __init__(
        self,
        snapshot_path: str,
        device: str = "cuda",
        keep_ratio: float = 0.8,
        sample_threshold: int = 1,
    ):
        # -------- device --------
        if device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # -------- cfg --------
        self.cfg = make_cfg()

        # -------- model --------
        self.model = create_model(self.cfg).to(self.device)
        state = torch.load(snapshot_path, map_location="cpu")
        self.model.load_state_dict(state["model"], strict=True)
        self.model.eval()

        # -------- cache --------
        self.neighbor_limits = None
        self.keep_ratio = keep_ratio
        self.sample_threshold = sample_threshold

    # ------------------------------------------------------------------
    # calibration (只需要做一次)
    # ------------------------------------------------------------------
    def calibrate(self, ref_points: np.ndarray, src_points: np.ndarray):
        """
        计算 neighbor_limits（强烈建议只做一次）
        """
        identity = np.eye(4, dtype=np.float32)
        dataset = _SinglePairDataset(ref_points, src_points, identity)

        self.neighbor_limits = calibrate_neighbors_stack_mode(
            dataset,
            registration_collate_fn_stack_mode,
            self.cfg.backbone.num_stages,
            self.cfg.backbone.init_voxel_size,
            self.cfg.backbone.init_radius,
            keep_ratio=self.keep_ratio,
            sample_threshold=self.sample_threshold,
        )

        return self.neighbor_limits

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------
    def register(self, src_points: np.ndarray, ref_points: np.ndarray) -> np.ndarray:
        """
        输入:
            src_points: (N,3)
            ref_points: (M,3)
        输出:
            T_src_to_ref: (4,4)
        """
        if self.neighbor_limits is None:
            raise RuntimeError(
                "neighbor_limits is None. "
                "Call calibrate() once before inference."
            )

        identity = np.eye(4, dtype=np.float32)

        dataset = _SinglePairDataset(ref_points, src_points, identity)
        batch = registration_collate_fn_stack_mode(
            [dataset[0]],
            self.cfg.backbone.num_stages,
            self.cfg.backbone.init_voxel_size,
            self.cfg.backbone.init_radius,
            self.neighbor_limits,
            precompute_data=True,
        )

        if self.device.type == "cuda":
            batch = to_cuda(batch)

        with torch.no_grad():
            output = self.model(batch)

        est_transform = output["estimated_transform"]
        if est_transform.dim() == 3:
            est_transform = est_transform[0]

        return est_transform.detach().cpu().numpy()

    # ------------------------------------------------------------------
    # optional utils
    # ------------------------------------------------------------------
    def save_neighbor_limits(self, path: str):
        if self.neighbor_limits is None:
            raise RuntimeError("neighbor_limits is None.")
        np.save(path, np.asarray(self.neighbor_limits, dtype=np.int32))

    def load_neighbor_limits(self, path: str):
        self.neighbor_limits = np.load(path).tolist()


def uniform_sampling(points, num_samples):
    """均匀采样"""
    N = points.shape[0]
    if N <= num_samples:
        return points
    indices = np.random.choice(N, num_samples, replace=False)
    return points[indices]


if __name__ == "__main__":
    infer = GeoTransformerInfer(
    snapshot_path="../../weights/geotransformer-modelnet.pth.tar",
    device="cuda",          # or "cpu"
    )

    # 准备点云数据
    ref_points = np.loadtxt(r"D:\onedrive\Beiren-12.25-6.25\焊接点云数据\demo测试数据\F-0-crop.txt")[:, :3]    # (M,3)
    src_points = np.loadtxt(r"D:\onedrive\Beiren-12.25-6.25\焊接点云数据\demo测试数据\F-1-crop.txt")[:, :3]    # (N,3)
    
    # 下采样
    ref_idx = uniform_sampling(ref_points, 3000)
    src_idx = uniform_sampling(src_points, 3000)
    # ref_points = ref_points[ref_idx]
    # src_points = src_points[src_idx]
    print (f"ref points: {ref_points.shape}, src points: {src_points.shape}")
    # exit()
    
    # 第一次，用一组代表性点云做一次校准
    infer.calibrate(ref_points, src_points)

    # 后面高频调用
    T = infer.register(src_points, ref_points)   # src -> target
    
    # print raw matrix values
    for row in T:
        print(' '.join([f'{val:.6f}' for val in row]))