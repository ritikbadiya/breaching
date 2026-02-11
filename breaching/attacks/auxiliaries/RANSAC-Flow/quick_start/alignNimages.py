import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)
sys.path.append(os.path.join(SCRIPT_DIR, '..', 'utils'))
sys.path.append(os.path.join(SCRIPT_DIR, '..', 'model'))

# import
from coarseAlignFeatMatch import CoarseAlign
import outil
import model as model
import PIL.Image as Image
import numpy as np
import torch
from torchvision import transforms
import argparse
import warnings
import torch.nn.functional as F
import kornia.geometry as tgm

if not sys.warnoptions:
    warnings.simplefilter("ignore")


def _load_networks(resume_pth, kernel_size):
    Transform = outil.Homography
    network = {
        'netFeatCoarse': model.FeatureExtractor(),
        'netCorr': model.CorrNeigh(kernel_size),
        'netFlowCoarse': model.NetFlowCoarse(kernel_size),
        'netMatch': model.NetMatchability(kernel_size),
    }
    for key in list(network.keys()):
        network[key].cuda()

    param = torch.load(resume_pth)
    for key in list(param.keys()):
        network[key].load_state_dict(param[key])
        network[key].eval()

    return Transform, network


def align_sources_to_target_images(
    source_images,
    target_img,
    args,
    network=None,
    save=False,
    output_names=None,
    imageNet=True,
    coarse_model=None,
):
    # Load networks (optional reuse)
    if network is None:
        _, network = _load_networks(args.resumePth, args.kernelSize)

    # Coarse alignment model (target is fixed)
    if coarse_model is None:
        coarse_model = CoarseAlign(
            args.nbScale,
            args.coarseIter,
            args.coarsetolerance,
            'Homography',
            args.minSize,
            segId=1,
            segFg=True,
            imageNet=imageNet,
            scaleR=args.scaleR,
            mean=getattr(args, "mean", None),
            std=getattr(args, "std", None),
        )
    coarse_model.setTarget(target_img)

    img2w, img2h = coarse_model.It.size
    gridX = torch.linspace(-1, 1, steps=img2w).view(1, 1, -1, 1).expand(1, img2h, img2w, 1)
    gridY = torch.linspace(-1, 1, steps=img2h).view(1, -1, 1, 1).expand(1, img2h, img2w, 1)
    grid = torch.cat((gridX, gridY), dim=3).cuda()
    warper = tgm.HomographyWarper(img2h, img2w)

    if save and args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
        coarse_model.It.save(os.path.join(args.outdir, 'resized_target.png'))

    aligned_images = []
    for idx, src_img in enumerate(source_images):
        coarse_model.setSource(src_img)

        best_prm, _ = coarse_model.getCoarse(np.zeros((img2h, img2w)))
        if best_prm is None:
            # Fallback: return resized source if coarse alignment fails
            aligned_images.append(coarse_model.Is)
            continue

        best_prm = torch.from_numpy(best_prm).unsqueeze(0).cuda()
        warper.precompute_warp_grid(best_prm)
        flow_coarse = warper._warped_grid
        img1_coarse = F.grid_sample(coarse_model.IsTensor, flow_coarse)

        # fine alignment
        feat1 = F.normalize(network['netFeatCoarse'](img1_coarse.cuda()))
        feat2 = F.normalize(network['netFeatCoarse'](coarse_model.ItTensor))
        corr12 = network['netCorr'](feat1, feat2)
        flow_down = network['netFlowCoarse'](corr12, False)
        flow_up = F.interpolate(flow_down, size=(grid.size()[1], grid.size()[2]), mode='bilinear')
        flow_up = flow_up.permute(0, 2, 3, 1)
        flow_up = flow_up + grid
        flow12 = F.grid_sample(flow_coarse.permute(0, 3, 1, 2), flow_up).permute(0, 2, 3, 1).contiguous()

        img1_fine = F.grid_sample(coarse_model.IsTensor, flow12)
        img1_fine_pil = transforms.ToPILImage()(img1_fine.cpu().squeeze())

        aligned_images.append(img1_fine_pil)

        if save and args.outdir:
            if output_names is not None and idx < len(output_names):
                out_name = output_names[idx]
            else:
                out_name = f'aligned_{idx}.png'
            out_path = os.path.join(args.outdir, out_name)
            img1_fine_pil.save(out_path)

    return aligned_images


def align_sources_to_target(source_paths, target_path, args):
    # Load target image
    target_img = Image.open(target_path).convert('RGB')
    source_images = [Image.open(p).convert('RGB') for p in source_paths]
    output_names = [f"aligned_{os.path.splitext(os.path.basename(p))[0]}.png" for p in source_paths]
    aligned_images = align_sources_to_target_images(
        source_images,
        target_img,
        args,
        network=None,
        save=True,
        output_names=output_names,
    )
    return aligned_images


def _parse_source_list(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def alignNimages(args):
    if args.source_list:
        sources = _parse_source_list(args.source_list)
    else:
        sources = args.sources

    if not sources:
        raise ValueError("No source images provided. Use --sources or --source_list.")

    return align_sources_to_target(sources, args.target, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Align multiple images to a single target')
    parser.add_argument('--target', type=str, required=True, help='path to the target image')
    parser.add_argument('--sources', type=str, nargs='+', help='list of source image paths')
    parser.add_argument('--source_list', type=str, help='text file with one source path per line')
    parser.add_argument('--outdir', type=str, help='path to the output folder', default='../output/')
    parser.add_argument('--resumePth', type=str, help='path to the model', default='../model/pretrained/MegaDepth_Theta1_Eta001_Grad1_0.774.pth')
    parser.add_argument('--kernelSize', type=int, help='size of the kernel', default=7)
    parser.add_argument('--nbPoint', type=int, help='number of points to use for alignment', default=4)
    parser.add_argument('--nbScale', type=int, help='number of scales to use for alignment', default=7)
    parser.add_argument('--coarseIter', type=int, help='number of iterations for coarse alignment', default=10000)
    parser.add_argument('--coarsetolerance', type=float, help='tolerance for coarse alignment', default=0.05)
    parser.add_argument('--minSize', type=int, help='minimum size for the image', default=400)
    parser.add_argument('--scaleR', type=float, help='scale ratio', default=1.2)

    args = parser.parse_args()
    alignNimages(args)
