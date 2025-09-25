import mcc_model
import util.misc as misc
import cv2
from pytorch3d.io.obj_io import load_obj
import torch


def call_mcc(image, point_cloud, args):

    # Get model
    model = mcc_model.get_mcc_model(
        occupancy_weight=1.0,
        rgb_weight=0.01,
        args=args,
    ).cuda()

    misc.load_model(args=args, model_without_ddp=model, optimizer=None, loss_scaler=None)

    # Take inputs
    rgb = cv2.imread(image)

    # Process input
    # convert numpy to correct type (6-element tuple with FloatTensor (V, 3), faces, aux, 
    #obj = load_obj(point_cloud)

    seen_rgb = (torch.tensor(rgb).float() / 255)[..., [2, 1, 0]]
    H, W = seen_rgb.shape[:2]
    seen_rgb = torch.nn.functional.interpolate(
        seen_rgb.permute(2, 0, 1)[None],
        size=[H, W],
        mode="bilinear",
        align_corners=False,
    )[0].permute(1, 2, 0)

    seen_xyz = torch.from_numpy(point_cloud).to(torch.float32).reshape(H, W, 3) #obj[0].reshape(H, W, 3)
    seg = cv2.imread(args.seg, cv2.IMREAD_UNCHANGED)
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

    # Samples?
    # Processing

    # Run prediction
    model.eval()

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
