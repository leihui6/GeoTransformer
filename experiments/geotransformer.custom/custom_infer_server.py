'''
用于推理的自定义GeoTransformer服务器
'''

import socket
import json
import mmap
import numpy as np
import torch
import os.path as osp
import open3d as o3d

SHM_NAME = "PointCloudSHM"
SHM_SIZE = 1024 * 1024 * 1024  # 1GB

from config import make_cfg
from geotransformer.utils.data import (
    registration_collate_fn_stack_mode,
    calibrate_neighbors_stack_mode,
)
from custom_infer import build_infer_data_dict, SinglePairDataset, downsample_points
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

def points_scale(src, ref, scale = 5.0):
    src_scaled = src * scale
    ref_scaled = ref * scale
    return src_scaled, ref_scaled

def recover_T_from_scaled(T_scaled, scale):
    T_recovered = np.eye(4, dtype=np.float32)
    T_recovered[:3, :3] = T_scaled[:3, :3]
    T_recovered[:3, 3] = T_scaled[:3, 3] / scale
    return T_recovered

def show_pointcloud(src, ref, tsfm=None, title="Point Clouds"):
    src_pcd = o3d.geometry.PointCloud()
    ref_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(src)
    if tsfm is not None:
        src_pcd.transform(tsfm)
    ref_pcd.points = o3d.utility.Vector3dVector(ref)
    src_pcd.paint_uniform_color([0, 1, 0])  # 绿色
    ref_pcd.paint_uniform_color([1, 0, 0])  # 红色
    o3d.visualization.draw_geometries([src_pcd, ref_pcd], 
                                      window_name=title, 
                                      width=800, height=600)

if __name__ == "__main__":
    USE_GPU = True
    SNAPSHOT_PATH = osp.join('./weights', 'epoch-35.pth.tar')
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
    sock.bind(("127.0.0.1", 60000))
    sock.listen(1)
    print("[Python] Waiting for C++...")
    conn, _ = sock.accept()
    print("[Python] Connected")

    o_src_points_np = None
    o_ref_points_np = None

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
            o_src_points_np = (
                np.frombuffer(mm, dtype=np.float32, 
                              count=num_points * 3, offset=0)
                .reshape(num_points, 3)
                .copy()
            )
            # show_pointcloud(o_src_points_np, np.array([[0,0,0]]))
            # print (id(o_src_points_np))
            print (f"[Python] Read {o_src_points_np.shape[0]} points from shared memory")
        elif msg["data_name"] == 'tgt':
            o_ref_points_np = (
                np.frombuffer(mm, dtype=np.float32, 
                              count=num_points * 3, offset=o_src_points_np.nbytes)
                .reshape(num_points, 3)
                .copy()
            )
            # print (id(o_ref_points_np))
            print (f"[Python] Read {o_ref_points_np.shape[0]} points from shared memory")

        if o_src_points_np is not None and o_ref_points_np is not None:
            print ("[Python] Both point clouds received, performing inference...")
            # show_pointcloud(o_src_points_np, o_ref_points_np, title="Received Point Clouds")
            # np.savetxt('debug_o_ref_points.txt', o_ref_points_np, fmt='%.6f')
            # np.savetxt('debug_o_src_points.txt', o_src_points_np, fmt='%.6f')
            # exit()
            # 放大点云
            src_points_np, ref_points_np = points_scale(o_src_points_np, o_ref_points_np, scale = 5.0)
            # 点云采样
            src_points_np = downsample_points(src_points_np, 6000)
            ref_points_np = downsample_points(ref_points_np, 6000)
            
            neighbor_limits = make_neighbor_limits(
                NEIGHBORSFILE="T4_neighbor_limits.txt",
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
            T_est = recover_T_from_scaled(T_est.squeeze(0).detach().cpu().numpy(), scale=5.0)
            np.set_printoptions(precision=6, suppress=True)
            print (f"[Python] Estimated Transformation:\n{T_est}")
            
            # === 写回共享内存 ===
            tsfm_np = np.ndarray(
                shape=(16,),
                dtype=np.float32,
                buffer=mm,
                offset=o_src_points_np.nbytes + o_ref_points_np.nbytes
            )
            np.copyto(tsfm_np, T_est.reshape(-1))
            print(f"[Python] Registration result written to shared memory")
            
            # === 回执 ===
            reply = "DONE"
            conn.send(json.dumps(reply).encode("utf-8"))
            print("[Python] DONE sent")
            
            # 用open3d 可视化出来
            show_pointcloud(o_src_points_np, o_ref_points_np, tsfm=T_est, title="Aligned Point Clouds")
            
            o_src_points_np = None
            o_ref_points_np = None