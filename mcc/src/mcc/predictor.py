# src/mcc/predictor.py
from typing import Dict
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

class MCCPredictor:
    def __init__(self, model):
        self.model = model.eval()
        self.device = next(model.parameters()).device

        # Read config from the model 
        self.granularity   = getattr(self.model, "granularity", 0.05)
        self.regress_color = getattr(self.model, "regress_color", False)
        self.temperature   = getattr(self.model, "temperature", 0.1)
        self.score_thresholds = getattr(self.model, "score_thresholds", [0.3])
        self.query_volume = getattr(self.model, "query_volume", 3.0)

    def predict(self, 
                point_cloud: np.ndarray,
                rgb: np.ndarray
        ) -> Dict[str, np.ndarray]:

        print("Preparing inputs...")
        # Input RGB image
        seen_rgb = (torch.tensor(rgb).float() / 255)[..., [2, 1, 0]]
        H, W = seen_rgb.shape[:2]

        seen_rgb = torch.nn.functional.interpolate(
            seen_rgb.permute(2, 0, 1)[None],
            size=[H, W],
            mode="bilinear",
            align_corners=False,
        )[0].permute(1, 2, 0)

        # Unprojected xyz points
        seen_xyz = torch.from_numpy(point_cloud).float().reshape(H, W, 3)                # [H,W,3]

        # Normalize

        seen_xyz = self._normalize(seen_xyz)

        # --- pad & resize  ---
        seen_xyz = self._pad_image(seen_xyz, float("inf"))
        seen_rgb = self._pad_image(seen_rgb, 0)
    
        seen_rgb = F.interpolate(
            seen_rgb.permute(2, 0, 1)[None],
            size=[800, 800],
            mode="bilinear",
            align_corners=False,
        )

        seen_xyz = F.interpolate(
            seen_xyz.permute(2, 0, 1)[None],
            size=[112, 112],
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1)

        # Send inputs to device
        seen_rgb = seen_rgb.to(self.device)
        seen_xyz = seen_xyz.to(self.device)

        print(f"preparing data; seen_rgb shape: {seen_rgb.shape}, seen_xyz shape: {seen_xyz.shape}")
        # Prepare data
        seen_xyz, valid_seen_xyz, unseen_xyz, unseen_rgb, seen_images = self._prepare_data(
            seen_xyz, seen_rgb
        )

        pred_occupy = []
        pred_color = []

        max_n_unseen_fwd = 2000

        print("predicting")
        self.model.cached_enc_feat = None
        num_passes = int(np.ceil(unseen_xyz.shape[1] / max_n_unseen_fwd))
        print(f'num_passes: {num_passes}')
        print(f"starting loop for {num_passes} passes")
        for p_idx in tqdm(range(num_passes)):
            print(f"pass {p_idx+1}")
            p_start = p_idx     * max_n_unseen_fwd
            p_end = (p_idx + 1) * max_n_unseen_fwd
            cur_unseen_xyz = unseen_xyz[:, p_start:p_end]
            cur_unseen_rgb = unseen_rgb[:, p_start:p_end].zero_()

            print(f"cur_unseen_xyz shape: {cur_unseen_xyz.shape}")
            print("model forward")

            with torch.inference_mode():
                _, pred = self.model(
                    seen_images=seen_images,
                    seen_xyz=seen_xyz,
                    unseen_xyz=cur_unseen_xyz,
                    unseen_rgb=cur_unseen_rgb,
                    cache_enc=True,
                    valid_seen_xyz=valid_seen_xyz,
                )

            print("model forward done")

            pred_occupy.append(pred[..., 0].cpu())
            if self.regress_color:
                pred_color.append(pred[..., 1:].reshape((-1, 3)))
            else:
                pred_color.append(
                    (
                        torch.nn.Softmax(dim=2)(
                            pred[..., 1:].reshape((-1, 3, 256)) / self.temperature
                        ) * torch.linspace(0, 1, 256, device=pred.device)
                    ).sum(axis=2)
                )

        # Output
        print("preparing output")
        clouds = self._generate_output(
            torch.cat(pred_occupy, dim=1),
            torch.cat(pred_color, dim=0),
            unseen_xyz,
            self.score_thresholds
        )

    # --- helpers ---
    def _pad_image(self, im, value):
        if im.shape[0] > im.shape[1]:
            diff = im.shape[0] - im.shape[1]
            return torch.cat([im, (torch.zeros((im.shape[0], diff, im.shape[2]), device=im.device) + value)], dim=1)
        else:
            diff = im.shape[1] - im.shape[0]
            return torch.cat([im, (torch.zeros((diff, im.shape[1], im.shape[2]), device=im.device) + value)], dim=0)

    def _normalize(self, seen_xyz):
        seen_xyz = seen_xyz / (seen_xyz[torch.isfinite(seen_xyz.sum(dim=-1))].var(dim=0) ** 0.5).mean()
        seen_xyz = seen_xyz - seen_xyz[torch.isfinite(seen_xyz.sum(dim=-1))].mean(axis=0)
        return seen_xyz
    
    def _prepare_data(self, seen_xyz: torch.Tensor, seen_rgb: torch.Tensor):
        device = seen_xyz.device
        print(f"shape seen_xyz: {seen_xyz.shape}, seen_rgb: {seen_rgb.shape}")
        #  Seen & valid mask 
        valid_seen_xyz = torch.isfinite(seen_xyz.sum(dim=-1))          
        seen_xyz = seen_xyz.clone()
        seen_xyz[~valid_seen_xyz] = -100                              

        # Build unseen queries 
        B = 1  # batch size
        unseen_xyz, unseen_rgb = self._get_grid(B)
                                                         # [1,M,3]
        # seen_images is just the image tensor the model expects
        seen_images = seen_rgb

        return seen_xyz, valid_seen_xyz, unseen_xyz, unseen_rgb, seen_images

    def _get_grid(self, B):
        W = float(self.query_volume)
        N = int(np.ceil(2 * W / float(self.granularity)))

        coords = torch.arange(N, device=self.device)

        a = (coords - (N / 2.0)) / (N / 2.0)
        axis = a * W

        x, y, z = torch.meshgrid(axis, axis, axis, indexing='ij')
        grid = torch.stack([x, y, z], dim=-1).reshape(1, -1, 3)  # [1, N^3, 3]
        unseen_xyz = grid.repeat(B, 1, 1) if B > 1 else grid.to(self.device)
        unseen_rgb = torch.zeros_like(unseen_xyz)
        
        return unseen_xyz, unseen_rgb


    def _generate_output(self,
                         pred_occ, 
                         pred_rgb, 
                         unseen_xyz,
                         score_thresholds=[0.3]):
        
        clouds = {'points': None}

        for t in score_thresholds:
            pos = pred_occ > t

            pts = unseen_xyz[pos]
            cols = pred_rgb[pos]

            good = pts[:, 0] != -100
            if good.sum() == 0:
                continue

            clouds['points'] = {
                "xyz": pts[good].numpy().astype(np.float32),
                "colors": cols[good].numpy().astype(np.float32)
            }
            
        return clouds