'''
用于推理的自定义GeoTransformer服务器
'''

import socket
import json
import mmap
import numpy as np
import torch
import os.path as osp

SHM_NAME = "PointCloudSHM"
SHM_SIZE = 1024 * 1024 * 1024  # 1GB

from config import make_cfg
from geotransformer.utils.data import (
    registration_collate_fn_stack_mode,
    calibrate_neighbors_stack_mode,
)
from custom_infer import build_infer_data_dict, SinglePairDataset
from model import create_model

def make_neighbor_limits(NEIGHBORSFILE, ref_points, src_points, cfg):
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
    return neighbor_limits

def points_scale(points, scale = 3.0):
    centroid = np.mean(points, axis=0)
    points -= centroid
    points /= scale
    points += centroid
    return points, scale, centroid

def recover_T_from_scaled(T_scaled, scale, centroid):
    T_orig = np.eye(4)
    R = T_scaled[:3, :3]
    t = T_scaled[:3, 3]

    I = np.eye(3)
    t_orig = scale * t + (1.0 - scale) * (I - R) @ centroid

    T_orig[:3, :3] = R
    T_orig[:3, 3] = t_orig
    return T_orig

if __name__ == "__main__":
    USE_GPU = True
    SNAPSHOT_PATH = osp.join('./weights', 'epoch-8.pth.tar')
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        if USE_GPU and not torch.cuda.is_available():
            print('Warning: CUDA requested but not available. Falling back to CPU.')
        device = torch.device('cpu')
    cfg = make_cfg()
    model = create_model(cfg).to(device)
    state_dict = torch.load(SNAPSHOT_PATH, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict['model'], strict=True)
    print (f'Loading model from {SNAPSHOT_PATH}...')
    
    mm = mmap.mmap(
        -1,
        SHM_SIZE,
        tagname=SHM_NAME,
        access=mmap.ACCESS_WRITE
    )

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 9000))
    sock.listen(1)
    print("[Python] Waiting for C++...")
    conn, _ = sock.accept()
    print("[Python] Connected")

    src_points_np = None
    ref_points_np = None

    while True:
        print ("[Python] Waiting for command...")
        data = conn.recv(4096)
        if not data:
            break
        try:
            msg = json.loads(data.decode("utf-8"))
        except Exception as e:      
            print(f"[Python] Failed to parse JSON: {e}")
            continue
        print (f"[Python] Received message: {msg}")
        
        if msg["cmd"] == "QUIT":
            print ("[Python] QUIT received, exiting...")
            conn.close()
            break
        elif msg["cmd"] == "PAUSE":
            print ("[Python] PAUSE received, continuing...")
            continue
        elif msg["cmd"] != "RUN":
            continue
        num_points = int (msg["num_points"])
        print(f"[Python] RUN received, points = {num_points}")

        if msg["data_name"] == 'src':
            src_points_np = np.ndarray(
                shape=(num_points, 3),
                dtype=np.float32,
                buffer=mm
            )
            print (f"[Python] Read {src_points_np.shape[0]} points from shared memory")
        elif msg["data_name"] == 'tgt':
            ref_points_np = np.ndarray(
                shape=(num_points, 3),
                dtype=np.float32,
                buffer=mm
            )
            print (f"[Python] Read {ref_points_np.shape[0]} points from shared memory")

        if src_points_np is not None and ref_points_np is not None:
            print ("[Python] Both point clouds received, performing inference...")
            src_points_np, scale, centroid = points_scale(src_points_np)
            ref_points_np, _, _ = points_scale(ref_points_np)
            
            neighbor_limits = make_neighbor_limits(
                NEIGHBORSFILE="neighbor_limits.txt",
                ref_points=ref_points_np,
                src_points=src_points_np,
                cfg=cfg
            )
            
            data_dict = build_infer_data_dict(
                ref_points_np, src_points_np, 
                cfg=cfg, neighbor_limits=neighbor_limits, 
                device=device
            )
            print ("[Python] Data dict built, running model inference...")
            model.eval()
            with torch.no_grad():
                T_est = model.forward_infer(data_dict)
            print ("[Python] Model inference done.")
            T_est = recover_T_from_scaled(T_est.squeeze(0).detach().cpu().numpy(), scale, centroid)
            
            # === 写回共享内存 ===
            tsfm_np = np.ndarray(
                shape=(16,),
                dtype=np.float32,
                buffer=mm,
            )
            np.copyto(tsfm_np, T_est.reshape(-1))
            print(f"[Python] Registration result written to shared memory")
            src_points_np = None
            ref_points_np = None
            # === 回执（关键）===
            reply = "DONE"
            conn.send(json.dumps(reply).encode("utf-8"))
            print("[Python] DONE sent")