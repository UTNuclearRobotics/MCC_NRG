import mcc_model
import util.misc as misc
import cv2
from pytorch3d.io.obj_io import load_obj
import torch
from tqdm import tqdm
from engine_mcc import prepare_data, generate_html
import numpy as np
import main_mcc

def generate_output(pred_occ, 
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

def run_viz(model, samples, device, args, prefix):
    model.eval()

    seen_xyz, valid_seen_xyz, unseen_xyz, unseen_rgb, labels, seen_images = prepare_data(
        samples, device, is_train=False, args=args, is_viz=True
    )
    pred_occupy = []
    pred_colors = []

    max_n_unseen_fwd = 2000

    model.cached_enc_feat = None
    num_passes = int(np.ceil(unseen_xyz.shape[1] / max_n_unseen_fwd))
    for p_idx in tqdm(range(num_passes)):
        p_start = p_idx     * max_n_unseen_fwd
        p_end = (p_idx + 1) * max_n_unseen_fwd
        cur_unseen_xyz = unseen_xyz[:, p_start:p_end]
        cur_unseen_rgb = unseen_rgb[:, p_start:p_end].zero_()
        cur_labels = labels[:, p_start:p_end].zero_()

        with torch.no_grad():
            _, pred = model(
                seen_images=seen_images,
                seen_xyz=seen_xyz,
                unseen_xyz=cur_unseen_xyz,
                unseen_rgb=cur_unseen_rgb,
                unseen_occupy=cur_labels,
                cache_enc=True,
                valid_seen_xyz=valid_seen_xyz,
            )
        pred_occupy.append(pred[..., 0].cpu())
        if args.regress_color:
            pred_colors.append(pred[..., 1:].reshape((-1, 3)))
        else:
            pred_colors.append(
                (
                    torch.nn.Softmax(dim=2)(
                        pred[..., 1:].reshape((-1, 3, 256)) / args.temperature
                    ) * torch.linspace(0, 1, 256, device=pred.device)
                ).sum(axis=2)
            )

    return generate_output(torch.cat(pred_occupy, dim=1),
                           torch.cat(pred_colors, dim=0),
                           unseen_xyz, 
                           score_thresholds=[0.3]
            )

def make_args(
        granularity=0.05,
        score_thresholds=[0.3],
        temperature=0.1,
        checkpoint='co3dv2_all_categories.pth',
        output=None
    ):

    parser = main_mcc.get_args_parser()
    parser.add_argument('--granularity', default=0.05, type=float, help='output granularity')
    parser.add_argument('--score_thresholds', default=[0.3], type=float, nargs='+', help='score thresholds')
    parser.add_argument('--temperature', default=0.1, type=float, help='temperature for color prediction.')
    parser.add_argument('--checkpoint', default='co3dv2_all_categories.pth', type=str, help='model checkpoint')

    parser.set_defaults(eval=True)
    args = parser.parse_args()

    args.granularity = granularity
    args.viz_granularity = granularity
    args.score_thresholds = score_thresholds
    args.temperature = temperature
    args.checkpoint = checkpoint
    args.resume = checkpoint
    args.output = output

    return args

def normalize(seen_xyz):
    seen_xyz = seen_xyz / (seen_xyz[torch.isfinite(seen_xyz.sum(dim=-1))].var(dim=0) ** 0.5).mean()
    seen_xyz = seen_xyz - seen_xyz[torch.isfinite(seen_xyz.sum(dim=-1))].mean(axis=0)
    return seen_xyz

def pad_image(im, value):
    if im.shape[0] > im.shape[1]:
        diff = im.shape[0] - im.shape[1]
        return torch.cat([im, (torch.zeros((im.shape[0], diff, im.shape[2])) + value)], dim=1)
    else:
        diff = im.shape[1] - im.shape[0]
        return torch.cat([im, (torch.zeros((diff, im.shape[1], im.shape[2])) + value)], dim=0)

def call_mcc(image, 
             point_cloud, 
             seg=None,
             granularity=0.05, 
             score_thresholds=[0.3],
             temperature=0.1,
             checkpoint='co3dv2_all_categories.pth',
             outputs=None
    ):
    # Create args
    args = make_args(granularity, score_thresholds, temperature, checkpoint, outputs)
    
    # Get model
    model = mcc_model.get_mcc_model(
        occupancy_weight=1.0,
        rgb_weight=0.01,
        args=args,
    ).cuda()

    misc.load_model(args=args, model_without_ddp=model, optimizer=None, loss_scaler=None)

    # Process inputs
    rgb = image 
    seen_rgb = (torch.tensor(rgb).float() / 255)[..., [2, 1, 0]]
    H, W = seen_rgb.shape[:2]

    seen_rgb = torch.nn.functional.interpolate(
        seen_rgb.permute(2, 0, 1)[None],
        size=[H, W],
        mode="bilinear",
        align_corners=False,
    )[0].permute(1, 2, 0)

    seen_xyz = torch.from_numpy(point_cloud).to(torch.float32).reshape(H, W, 3) #obj[0].reshape(H, W, 3)
    
    # Check for segmentation image
    if seg is not None:
        mask = torch.tensor(cv2.resize(seg, (W, H))).bool()
    
        seen_xyz[~mask] = float('inf')

        seen_xyz = normalize(seen_xyz)

        bottom, right = mask.nonzero().max(dim=0)[0]
        top, left = mask.nonzero().min(dim=0)[0]

        bottom = bottom + 40
        right = right + 40
        top = max(top - 40, 0)
        left = max(left - 40, 0)

        seen_xyz = seen_xyz[top:bottom+1, left:right+1]
        seen_rgb = seen_rgb[top:bottom+1, left:right+1]

    seen_xyz = pad_image(seen_xyz, float('inf'))
    seen_rgb = pad_image(seen_rgb, 0)

    seen_rgb = torch.nn.functional.interpolate(
        seen_rgb.permute(2, 0, 1)[None],
        size=[800, 800],
        mode="bilinear",
        align_corners=False,
    )

    seen_xyz = torch.nn.functional.interpolate(
        seen_xyz.permute(2, 0, 1)[None],
        size=[112, 112],
        mode="bilinear",
        align_corners=False,
    ).permute(0, 2, 3, 1)

    # Create pseudo-batch
    sample = [
        [seen_xyz, seen_rgb],
        [torch.zeros((20000, 3)), torch.zeros((20000, 3))],
    ]
    
    # Perform inference
    return run_viz(model, sample, "cuda", args, prefix=args.output)