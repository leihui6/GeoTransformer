import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed

from geotransformer.modules.ops import point_to_node_partition, index_select
from geotransformer.modules.registration import get_node_correspondences
from geotransformer.modules.sinkhorn import LearnableLogOptimalTransport
from geotransformer.modules.geotransformer import (
    GeometricTransformer,
    SuperPointMatching,
    SuperPointTargetGenerator,
    LocalGlobalRegistration,
)

from backbone import KPConvFPN


class GeoTransformer(nn.Module):
    def __init__(self, cfg):
        super(GeoTransformer, self).__init__()
        self.num_points_in_patch = cfg.model.num_points_in_patch
        self.matching_radius = cfg.model.ground_truth_matching_radius

        self.backbone = KPConvFPN(
            cfg.backbone.input_dim,
            cfg.backbone.output_dim,
            cfg.backbone.init_dim,
            cfg.backbone.kernel_size,
            cfg.backbone.init_radius,
            cfg.backbone.init_sigma,
            cfg.backbone.group_norm,
        )

        self.transformer = GeometricTransformer(
            cfg.geotransformer.input_dim,
            cfg.geotransformer.output_dim,
            cfg.geotransformer.hidden_dim,
            cfg.geotransformer.num_heads,
            cfg.geotransformer.blocks,
            cfg.geotransformer.sigma_d,
            cfg.geotransformer.sigma_a,
            cfg.geotransformer.angle_k,
            reduction_a=cfg.geotransformer.reduction_a,
        )

        self.coarse_target = SuperPointTargetGenerator(
            cfg.coarse_matching.num_targets, cfg.coarse_matching.overlap_threshold
        )

        self.coarse_matching = SuperPointMatching(
            cfg.coarse_matching.num_correspondences, cfg.coarse_matching.dual_normalization
        )

        self.fine_matching = LocalGlobalRegistration(
            cfg.fine_matching.topk,
            cfg.fine_matching.acceptance_radius,
            mutual=cfg.fine_matching.mutual,
            confidence_threshold=cfg.fine_matching.confidence_threshold,
            use_dustbin=cfg.fine_matching.use_dustbin,
            use_global_score=cfg.fine_matching.use_global_score,
            correspondence_threshold=cfg.fine_matching.correspondence_threshold,
            correspondence_limit=cfg.fine_matching.correspondence_limit,
            num_refinement_steps=cfg.fine_matching.num_refinement_steps,
        )

        self.optimal_transport = LearnableLogOptimalTransport(cfg.model.num_sinkhorn_iterations)

    def forward(self, data_dict):
        output_dict = {}

        # Downsample point clouds
        feats = data_dict['features'].detach()
        transform = data_dict['transform'].detach()

        ref_length_c = data_dict['lengths'][-1][0].item()
        ref_length_f = data_dict['lengths'][1][0].item()
        ref_length = data_dict['lengths'][0][0].item()
        points_c = data_dict['points'][-1].detach()
        points_f = data_dict['points'][1].detach()
        points = data_dict['points'][0].detach()

        ref_points_c = points_c[:ref_length_c]
        src_points_c = points_c[ref_length_c:]
        ref_points_f = points_f[:ref_length_f]
        src_points_f = points_f[ref_length_f:]
        ref_points = points[:ref_length]
        src_points = points[ref_length:]

        output_dict['ref_points_c'] = ref_points_c
        output_dict['src_points_c'] = src_points_c
        output_dict['ref_points_f'] = ref_points_f
        output_dict['src_points_f'] = src_points_f
        output_dict['ref_points'] = ref_points
        output_dict['src_points'] = src_points

        # 1. Generate ground truth node correspondences
        _, ref_node_masks, ref_node_knn_indices, ref_node_knn_masks = point_to_node_partition(
            ref_points_f, ref_points_c, self.num_points_in_patch
        )
        _, src_node_masks, src_node_knn_indices, src_node_knn_masks = point_to_node_partition(
            src_points_f, src_points_c, self.num_points_in_patch
        )

        ref_padded_points_f = torch.cat([ref_points_f, torch.zeros_like(ref_points_f[:1])], dim=0)
        src_padded_points_f = torch.cat([src_points_f, torch.zeros_like(src_points_f[:1])], dim=0)
        ref_node_knn_points = index_select(ref_padded_points_f, ref_node_knn_indices, dim=0)
        src_node_knn_points = index_select(src_padded_points_f, src_node_knn_indices, dim=0)

        gt_node_corr_indices, gt_node_corr_overlaps = get_node_correspondences(
            ref_points_c,
            src_points_c,
            ref_node_knn_points,
            src_node_knn_points,
            transform,
            self.matching_radius,
            ref_masks=ref_node_masks,
            src_masks=src_node_masks,
            ref_knn_masks=ref_node_knn_masks,
            src_knn_masks=src_node_knn_masks,
        )

        output_dict['gt_node_corr_indices'] = gt_node_corr_indices
        output_dict['gt_node_corr_overlaps'] = gt_node_corr_overlaps

        # 2. KPFCNN Encoder
        feats_list = self.backbone(feats, data_dict)

        feats_c = feats_list[-1]
        feats_f = feats_list[0]

        # 3. Conditional Transformer
        ref_feats_c = feats_c[:ref_length_c]
        src_feats_c = feats_c[ref_length_c:]
        ref_feats_c, src_feats_c = self.transformer(
            ref_points_c.unsqueeze(0),
            src_points_c.unsqueeze(0),
            ref_feats_c.unsqueeze(0),
            src_feats_c.unsqueeze(0),
        )
        ref_feats_c_norm = F.normalize(ref_feats_c.squeeze(0), p=2, dim=1)
        src_feats_c_norm = F.normalize(src_feats_c.squeeze(0), p=2, dim=1)

        output_dict['ref_feats_c'] = ref_feats_c_norm
        output_dict['src_feats_c'] = src_feats_c_norm

        # 5. Head for fine level matching
        ref_feats_f = feats_f[:ref_length_f]
        src_feats_f = feats_f[ref_length_f:]
        output_dict['ref_feats_f'] = ref_feats_f
        output_dict['src_feats_f'] = src_feats_f

        # 6. Select topk nearest node correspondences
        with torch.no_grad():
            ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_matching(
                ref_feats_c_norm, src_feats_c_norm, ref_node_masks, src_node_masks
            )

            output_dict['ref_node_corr_indices'] = ref_node_corr_indices
            output_dict['src_node_corr_indices'] = src_node_corr_indices

            # 7 Random select ground truth node correspondences during training
            if self.training:
                ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_target(
                    gt_node_corr_indices, gt_node_corr_overlaps
                )

        # 7.2 Generate batched node points & feats
        ref_node_corr_knn_indices = ref_node_knn_indices[ref_node_corr_indices]  # (P, K)
        src_node_corr_knn_indices = src_node_knn_indices[src_node_corr_indices]  # (P, K)
        ref_node_corr_knn_masks = ref_node_knn_masks[ref_node_corr_indices]  # (P, K)
        src_node_corr_knn_masks = src_node_knn_masks[src_node_corr_indices]  # (P, K)
        ref_node_corr_knn_points = ref_node_knn_points[ref_node_corr_indices]  # (P, K, 3)
        src_node_corr_knn_points = src_node_knn_points[src_node_corr_indices]  # (P, K, 3)

        ref_padded_feats_f = torch.cat([ref_feats_f, torch.zeros_like(ref_feats_f[:1])], dim=0)
        src_padded_feats_f = torch.cat([src_feats_f, torch.zeros_like(src_feats_f[:1])], dim=0)
        ref_node_corr_knn_feats = index_select(ref_padded_feats_f, ref_node_corr_knn_indices, dim=0)  # (P, K, C)
        src_node_corr_knn_feats = index_select(src_padded_feats_f, src_node_corr_knn_indices, dim=0)  # (P, K, C)

        output_dict['ref_node_corr_knn_points'] = ref_node_corr_knn_points
        output_dict['src_node_corr_knn_points'] = src_node_corr_knn_points
        output_dict['ref_node_corr_knn_masks'] = ref_node_corr_knn_masks
        output_dict['src_node_corr_knn_masks'] = src_node_corr_knn_masks

        # 8. Optimal transport
        matching_scores = torch.einsum('bnd,bmd->bnm', ref_node_corr_knn_feats, src_node_corr_knn_feats)  # (P, K, K)
        matching_scores = matching_scores / feats_f.shape[1] ** 0.5
        matching_scores = self.optimal_transport(matching_scores, ref_node_corr_knn_masks, src_node_corr_knn_masks)

        output_dict['matching_scores'] = matching_scores

        # 9. Generate final correspondences during testing
        with torch.no_grad():
            if not self.fine_matching.use_dustbin:
                matching_scores = matching_scores[:, :-1, :-1]

            ref_corr_points, src_corr_points, corr_scores, estimated_transform = self.fine_matching(
                ref_node_corr_knn_points,
                src_node_corr_knn_points,
                ref_node_corr_knn_masks,
                src_node_corr_knn_masks,
                matching_scores,
                node_corr_scores,
            )

            output_dict['ref_corr_points'] = ref_corr_points
            output_dict['src_corr_points'] = src_corr_points
            output_dict['corr_scores'] = corr_scores
            output_dict['estimated_transform'] = estimated_transform

        return output_dict
    
    # def forward_back(self, data_dict):
    #     feats = data_dict['features']
    #     ref_length_c = data_dict['lengths'][-1][0].item()
    #     ref_length_f = data_dict['lengths'][1][0].item()
    #     ref_length = data_dict['lengths'][0][0].item()

    #     points_c = data_dict['points'][-1]
    #     points_f = data_dict['points'][1]
    #     points = data_dict['points'][0]

    #     ref_points_c = points_c[:ref_length_c]
    #     src_points_c = points_c[ref_length_c:]
    #     ref_points_f = points_f[:ref_length_f]
    #     src_points_f = points_f[ref_length_f:]

    #     # 1. point → node partition
    #     _, ref_node_masks, ref_node_knn_indices, ref_node_knn_masks = point_to_node_partition(
    #         ref_points_f, ref_points_c, self.num_points_in_patch
    #     )
    #     _, src_node_masks, src_node_knn_indices, src_node_knn_masks = point_to_node_partition(
    #         src_points_f, src_points_c, self.num_points_in_patch
    #     )

    #     ref_padded_points_f = torch.cat([ref_points_f, torch.zeros_like(ref_points_f[:1])], dim=0)
    #     src_padded_points_f = torch.cat([src_points_f, torch.zeros_like(src_points_f[:1])], dim=0)
    #     ref_node_knn_points = index_select(ref_padded_points_f, ref_node_knn_indices, dim=0)
    #     src_node_knn_points = index_select(src_padded_points_f, src_node_knn_indices, dim=0)
        
    #     # 2. Backbone
    #     feats_list = self.backbone(feats, data_dict)
    #     feats_c = feats_list[-1]
    #     feats_f = feats_list[0]

    #     # 3. Transformer
    #     ref_feats_c = feats_c[:ref_length_c]
    #     src_feats_c = feats_c[ref_length_c:]

    #     ref_feats_c, src_feats_c = self.transformer(
    #         ref_points_c.unsqueeze(0),
    #         src_points_c.unsqueeze(0),
    #         ref_feats_c.unsqueeze(0),
    #         src_feats_c.unsqueeze(0),
    #     )

    #     ref_feats_c = F.normalize(ref_feats_c.squeeze(0), dim=1)
    #     src_feats_c = F.normalize(src_feats_c.squeeze(0), dim=1)

    #     # 4. Coarse matching
    #     ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_matching(
    #         ref_feats_c, src_feats_c, ref_node_masks, src_node_masks
    #     )

    #     # 5. Gather KNN patches
    #     ref_node_corr_knn_indices = ref_node_knn_indices[ref_node_corr_indices]
    #     src_node_corr_knn_indices = src_node_knn_indices[src_node_corr_indices]

    #     ref_node_corr_knn_masks = ref_node_knn_masks[ref_node_corr_indices]
    #     src_node_corr_knn_masks = src_node_knn_masks[src_node_corr_indices]

    #     ref_node_corr_knn_points = ref_node_knn_points[ref_node_corr_indices]
    #     src_node_corr_knn_points = src_node_knn_points[src_node_corr_indices]

    #     ref_feats_f = feats_f[:ref_length_f]
    #     src_feats_f = feats_f[ref_length_f:]

    #     ref_padded_feats_f = torch.cat([ref_feats_f, torch.zeros_like(ref_feats_f[:1])], dim=0)
    #     src_padded_feats_f = torch.cat([src_feats_f, torch.zeros_like(src_feats_f[:1])], dim=0)

    #     ref_node_corr_knn_feats = index_select(ref_padded_feats_f, ref_node_corr_knn_indices, dim=0)
    #     src_node_corr_knn_feats = index_select(src_padded_feats_f, src_node_corr_knn_indices, dim=0)

    #     # 6. Optimal transport
    #     matching_scores = torch.einsum(
    #         'bnd,bmd->bnm', ref_node_corr_knn_feats, src_node_corr_knn_feats
    #     )
    #     matching_scores = matching_scores / feats_f.shape[1] ** 0.5
    #     matching_scores = self.optimal_transport(
    #         matching_scores, ref_node_corr_knn_masks, src_node_corr_knn_masks
    #     )

    #     # if not self.fine_matching.use_dustbin:
    #     matching_scores = matching_scores[:, :-1, :-1]

    #     # 7. Fine matching & registration
    #     ref_corr_points, src_corr_points, corr_scores, estimated_transform = self.fine_matching(
    #         ref_node_corr_knn_points,
    #         src_node_corr_knn_points,
    #         ref_node_corr_knn_masks,
    #         src_node_corr_knn_masks,
    #         matching_scores,
    #         node_corr_scores,
    #     )

    #     # output_dict['estimated_transform'] = estimated_transform
    #     # output_dict['ref_corr_points'] = ref_corr_points
    #     # output_dict['src_corr_points'] = src_corr_points
    #     # output_dict['corr_scores'] = corr_scores
    #     output_dict = estimated_transform
    #     return output_dict

    # def forward(self, 
    #             feats, lengths, 
    #             # lengths_c, lengths_f, points_c, points_f,
    #             # ref_points_c, src_points_c, ref_points_f, src_points_f,
    #             points, neighbors, subsampling, upsampling):
    #     """
    #     data_dict 由外部 build_infer_data_dict 构造
    #     所有值都是 Tensor / list[Tensor]
    #     不做任何几何生成
    #     """
    #     # feats = data_dict['features']

    #     # # -------------------------------
    #     # # 1. 用 mask 代替 .item()
    #     # # -------------------------------
    #     # lengths_c = data_dict['lengths'][-1]   # (B,)
    #     # lengths_f = data_dict['lengths'][1]    # (B,)

    #     # points_c = data_dict['points'][-1]     # (Nc, 3) c -> corase points
    #     # points_f = data_dict['points'][1]      # (Nf, 3) f -> fine points

    #     # device = points_c.device

    #     # ref / src mask（B=2 时：[ref, src]）
    #     # ref_mask_c = torch.arange(points_c.shape[0], device=device) < lengths_c[0]
    #     # ref_mask_f = torch.arange(points_f.shape[0], device=device) < lengths_f[0]

    #     # ref_points_c = points_c[ref_mask_c]
    #     # src_points_c = points_c[~ref_mask_c]

    #     # ref_points_f = points_f[ref_mask_f]
    #     # src_points_f = points_f[~ref_mask_f]

    #     # lengths: tensor([ref_count, src_count])
    #     points_f = points[1]      # fine
    #     points_c = points[-1]     # coarse

    #     # 对应 lengths
    #     lengths_f = lengths[1]    # (2,)
    #     lengths_c = lengths[-1]   # (2,)

    #     # 拆分 fine
    #     ref_count_f = lengths_f[0]
    #     src_count_f = lengths_f[1]

    #     ref_points_f = torch.narrow(points_f, 0, 0, ref_count_f)
    #     src_points_f = torch.narrow(points_f, 0, ref_count_f, src_count_f)

    #     # 拆分 coarse
    #     ref_count_c = lengths_c[0]
    #     src_count_c = lengths_c[1]

    #     ref_points_c = torch.narrow(points_c, 0, 0, ref_count_c)
    #     src_points_c = torch.narrow(points_c, 0, ref_count_c, src_count_c)

    #     # print (f"ref_points_c shape: {ref_points_c.shape}, src_points_c shape: {src_points_c.shape}")
    #     # print (f"ref_points_f shape: {ref_points_f.shape}, src_points_f shape: {src_points_f.shape}")
    #     # exit()
    #     # -------------------------------
    #     # 2. point → node partition（保持不变）
    #     # -------------------------------
    #     _, ref_node_masks, ref_node_knn_indices, ref_node_knn_masks = point_to_node_partition(
    #         ref_points_f, ref_points_c, self.num_points_in_patch
    #     )
    #     _, src_node_masks, src_node_knn_indices, src_node_knn_masks = point_to_node_partition(
    #         src_points_f, src_points_c, self.num_points_in_patch
    #     )

    #     ref_padded_points_f = torch.cat(
    #         [ref_points_f, torch.zeros_like(ref_points_f[:1])], dim=0
    #     )
    #     src_padded_points_f = torch.cat(
    #         [src_points_f, torch.zeros_like(src_points_f[:1])], dim=0
    #     )

    #     ref_node_knn_points = index_select(
    #         ref_padded_points_f, ref_node_knn_indices, dim=0
    #     )
    #     src_node_knn_points = index_select(
    #         src_padded_points_f, src_node_knn_indices, dim=0
    #     )

    #     # -------------------------------
    #     # 3. Backbone（完全不动）
    #     # -------------------------------
    #     feats_list = self.backbone(feats, points, neighbors, subsampling, upsampling)
    #     feats_c = feats_list[-1]
    #     feats_f = feats_list[0]

    #     # ref_feats_c = feats_c[ref_mask_c]
    #     # src_feats_c = feats_c[~ref_mask_c]
        
    #     ref_feats_c = torch.narrow(feats_c, 0, 0, ref_count_c)
    #     src_feats_c = torch.narrow(feats_c, 0, ref_count_c, src_count_c)

    #     # -------------------------------
    #     # 4. Transformer
    #     # -------------------------------
    #     ref_feats_c, src_feats_c = self.transformer(
    #         ref_points_c.unsqueeze(0),
    #         src_points_c.unsqueeze(0),
    #         ref_feats_c.unsqueeze(0),
    #         src_feats_c.unsqueeze(0),
    #     )

    #     ref_feats_c = F.normalize(ref_feats_c.squeeze(0), dim=1)
    #     src_feats_c = F.normalize(src_feats_c.squeeze(0), dim=1)

    #     # # -------------------------------
    #     # # 5. Coarse matching
    #     # # -------------------------------
    #     ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_matching(
    #         ref_feats_c, src_feats_c, ref_node_masks, src_node_masks
    #     )

    #     # # -------------------------------
    #     # # 6. Gather KNN patches
    #     # # -------------------------------
    #     ref_node_corr_knn_indices = ref_node_knn_indices[ref_node_corr_indices]
    #     src_node_corr_knn_indices = src_node_knn_indices[src_node_corr_indices]

    #     ref_node_corr_knn_masks = ref_node_knn_masks[ref_node_corr_indices]
    #     src_node_corr_knn_masks = src_node_knn_masks[src_node_corr_indices]

    #     ref_node_corr_knn_points = ref_node_knn_points[ref_node_corr_indices]
    #     src_node_corr_knn_points = src_node_knn_points[src_node_corr_indices]

    #     ref_feats_f = torch.narrow(feats_f, 0, 0, ref_count_f)
    #     src_feats_f = torch.narrow(feats_f, 0, ref_count_f, src_count_f)

    #     ref_padded_feats_f = torch.cat(
    #         [ref_feats_f, torch.zeros_like(ref_feats_f[:1])], dim=0
    #     )
    #     src_padded_feats_f = torch.cat(
    #         [src_feats_f, torch.zeros_like(src_feats_f[:1])], dim=0
    #     )

    #     ref_node_corr_knn_feats = index_select(
    #         ref_padded_feats_f, ref_node_corr_knn_indices, dim=0
    #     )
    #     src_node_corr_knn_feats = index_select(
    #         src_padded_feats_f, src_node_corr_knn_indices, dim=0
    #     )

    #     # # -------------------------------
    #     # # 7. Optimal transport
    #     # # -------------------------------
    #     matching_scores = torch.einsum(
    #         'bnd,bmd->bnm', ref_node_corr_knn_feats, src_node_corr_knn_feats
    #     )
    #     matching_scores = matching_scores / feats_f.shape[1] ** 0.5

    #     matching_scores = self.optimal_transport(
    #         matching_scores, ref_node_corr_knn_masks, src_node_corr_knn_masks
    #     )

    #     matching_scores = matching_scores[:, :-1, :-1]

    #     # estimated_transform = torch.eye(4, device=device)  # 占位，保持输出格式正确，后续会被写回共享内存
    #     # print (matching_scores)
    #     # exit()
    #     # -------------------------------
    #     # 8. Fine matching & registration
    #     # -------------------------------
    #     _, _, _, estimated_transform = self.fine_matching(
    #         ref_node_corr_knn_points,
    #         src_node_corr_knn_points,
    #         ref_node_corr_knn_masks,
    #         src_node_corr_knn_masks,
    #         matching_scores,
    #         node_corr_scores,
    #     )

    #     return estimated_transform
    #     # return matching_scores


def create_model(config):
    model = GeoTransformer(config)
    return model


def main():
    from config import make_cfg

    cfg = make_cfg()
    model = create_model(cfg)
    print(model.state_dict().keys())
    print(model)


if __name__ == '__main__':
    main()
