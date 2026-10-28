import os
import argparse
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter
from collections import OrderedDict
import warnings
import matplotlib.pyplot as plt

# Import models/utils shared with inference.py
from models.siamese import Encoder, Predictor
from models.stn import stn_net
from losses.norm_loss import CosLoss
from utils.funcs import embedding_concat, mahalanobis_torch, rot_img, translation_img, hflip_img, rot90_img, grey_img

warnings.filterwarnings("ignore")

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

def load_image(image_path, img_size=224):
    """Load and preprocess a single image"""
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img)

def load_mask(mask_path, img_size=224):
    """Load ground truth mask"""
    if not os.path.exists(mask_path):
        return None
    
    mask = Image.open(mask_path).convert('L')
    mask = mask.resize((img_size, img_size), Image.NEAREST)
    mask = np.array(mask)
    
    # Convert to binary: 0 for normal, 1 for anomaly
    # Assuming standard datasets where 0 is background and >0 is anomaly
    mask = (mask > 0).astype(np.int32)
    return mask

def load_images_from_folder(folder_path, img_size=224):
    """Load all images from a folder"""
    images = []
    image_names = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    if not os.path.isdir(folder_path):
         raise ValueError(f"Directory not found: {folder_path}")

    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(valid_extensions):
            img_path = os.path.join(folder_path, filename)
            img_tensor = load_image(img_path, img_size)
            images.append(img_tensor)
            image_names.append(filename)

    if len(images) == 0:
        raise ValueError(f"No valid images found in {folder_path}")

    return torch.stack(images), image_names

# --- DATA AUGMENTATION (RegAD support adaptation) ---
def augment_support_images(support_img):
    augment_support_img = support_img
    # rotate
    for angle in [-np.pi/4, -3*np.pi/16, -np.pi/8, -np.pi/16, np.pi/16, np.pi/8, 3*np.pi/16, np.pi/4]:
        rotate_img_tensor = rot_img(support_img, angle)
        augment_support_img = torch.cat([augment_support_img, rotate_img_tensor], dim=0)
    # translate
    for a, b in [(0.2,0.2), (-0.2,0.2), (-0.2,-0.2), (0.2,-0.2), (0.1,0.1), (-0.1,0.1), (-0.1,-0.1), (0.1,-0.1)]:
        trans_img = translation_img(support_img, a, b)
        augment_support_img = torch.cat([augment_support_img, trans_img], dim=0)
    # hflip
    flipped_img = hflip_img(support_img)
    augment_support_img = torch.cat([augment_support_img, flipped_img], dim=0)
    # grey
    greyed_img = grey_img(support_img)
    augment_support_img = torch.cat([augment_support_img, greyed_img], dim=0)
    # rotate 90
    for angle in [1, 2, 3]:
        rotate90_img_tensor = rot90_img(support_img, angle)
        augment_support_img = torch.cat([augment_support_img, rotate90_img_tensor], dim=0)
    
    augment_support_img = augment_support_img[torch.randperm(augment_support_img.size(0))]
    return augment_support_img

# --- INFERENCE PIPELINE (RegAD support adaptation) ---
def get_score_map(args, models, support_imgs, test_imgs):
    STN = models[0]
    ENC = models[1]
    PRED = models[2]

    STN.eval()
    ENC.eval()
    PRED.eval()

    train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
    test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

    augment_support_img = augment_support_images(support_imgs)

    with torch.no_grad():
        support_feat = STN(augment_support_img.to(device))
    support_feat = torch.mean(support_feat, dim=0, keepdim=True)

    train_outputs['layer1'].append(STN.stn1_output)
    train_outputs['layer2'].append(STN.stn2_output)
    train_outputs['layer3'].append(STN.stn3_output)

    for k, v in train_outputs.items():
        train_outputs[k] = torch.cat(v, 0)

    embedding_vectors = train_outputs['layer1']
    for layer_name in ['layer2', 'layer3']:
        embedding_vectors = embedding_concat(embedding_vectors, train_outputs[layer_name], use_cuda)

    B, C, H, W = embedding_vectors.size()
    embedding_vectors = embedding_vectors.view(B, C, H * W)
    mean = torch.mean(embedding_vectors, dim=0)
    cov = torch.zeros(C, C, H * W).to(device)
    I = torch.eye(C).to(device)
    for i in range(H * W):
        cov[:, :, i] = torch.cov(embedding_vectors[:, :, i].T) + 0.01 * I
    train_outputs = [mean, cov]

    # Test
    for idx in range(test_imgs.size(0)):
        query_img = test_imgs[idx:idx+1].to(device)
        _ = STN(query_img) # Forward pass to store outputs
        test_outputs['layer1'].append(STN.stn1_output)
        test_outputs['layer2'].append(STN.stn2_output)
        test_outputs['layer3'].append(STN.stn3_output)

    for k, v in test_outputs.items():
        test_outputs[k] = torch.cat(v, 0)

    embedding_vectors = test_outputs['layer1']
    for layer_name in ['layer2', 'layer3']:
        embedding_vectors = embedding_concat(embedding_vectors, test_outputs[layer_name], use_cuda)

    B, C, H, W = embedding_vectors.size()
    embedding_vectors = embedding_vectors.view(B, C, H * W)
    dist_list = []

    for i in range(H * W):
        mean_val = train_outputs[0][:, i]
        conv_inv = torch.linalg.inv(train_outputs[1][:, :, i])
        dist = [mahalanobis_torch(sample[:, i], mean_val, conv_inv) for sample in embedding_vectors]
        dist_list.append(dist)

    dist_list = torch.tensor(dist_list).transpose(1, 0).reshape(B, H, W)

    # Upsample & Smooth (Crucial for consistency with visualizer)
    score_map = F.interpolate(dist_list.unsqueeze(1), size=args.img_size, mode='bilinear',
                             align_corners=False).squeeze(1).cpu().numpy()
    
    # Handle single image case (squeeze might remove batch dim if B=1)
    if score_map.ndim == 2:
        score_map = score_map[np.newaxis, :, :]

    for i in range(score_map.shape[0]):
        score_map[i] = gaussian_filter(score_map[i], sigma=4)

    return score_map

def main():
    parser = argparse.ArgumentParser(description='RegAD AUROC Evaluation')
    parser.add_argument('--obj', type=str, default='leather')
    parser.add_argument('--shot', type=int, default=2)
    parser.add_argument('--stn_mode', type=str, default='rotation_scale')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--support_dir', type=str, required=True, help='Support images dir')
    parser.add_argument('--test_dir', type=str, required=True, help='Test images dir')
    parser.add_argument('--mask_dir', type=str, required=True, help='Ground truth masks dir')
    parser.add_argument('--checkpoint_path', type=str, default=None)
    
    parser.add_argument('--output_dir', default='./outputs/evaluation')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    args.input_channel = 3

    print("="*60)
    print("RegAD AUROC Evaluation")
    print("="*60)

    # 1. Load Models
    print(f"\n1. Loading model ({args.obj}, shot={args.shot})...")
    STN = stn_net(args).to(device)
    ENC = Encoder().to(device)
    PRED = Predictor().to(device)

    # Determine checkpoint path
    if args.checkpoint_path:
        CKPT_name = args.checkpoint_path
        if not os.path.exists(CKPT_name):
            print(f"ERROR: Model checkpoint not found at specified path: {CKPT_name}")
            return
    else:
        # Try multiple possible checkpoint paths
        # Explicit or conventional checkpoint paths
        possible_paths = [
            f'./save_checkpoints/{args.stn_mode}/{args.shot}/{args.obj}/{args.obj}_{args.shot}_{args.stn_mode}_model.pt',
            f'./save_checkpoints/{args.shot}/{args.obj}/{args.obj}_{args.shot}_{args.stn_mode}_model.pt',
            f'./save_checkpoints/{args.obj}_{args.shot}_{args.stn_mode}_model.pt',
        ]

        CKPT_name = None
        for path in possible_paths:
            if os.path.exists(path):
                CKPT_name = path
                break

        if CKPT_name is None:
            print(f"ERROR: Model checkpoint not found in any of these locations:")
            for path in possible_paths:
                print(f"  - {path}")
            print("\nPlease use --checkpoint_path to specify the correct path.")
            return

    ckpt = torch.load(CKPT_name, map_location=device)
    STN.load_state_dict(ckpt['STN'])
    ENC.load_state_dict(ckpt['ENC'])
    PRED.load_state_dict(ckpt['PRED'])
    models = [STN, ENC, PRED]

    # 2. Load Data
    print(f"\n2. Loading data...")
    support_imgs, _ = load_images_from_folder(args.support_dir, args.img_size)
    test_imgs, test_names = load_images_from_folder(args.test_dir, args.img_size)
    print(f"   Support: {len(support_imgs)} images")
    print(f"   Test: {len(test_imgs)} images")

    # 3. Inference
    print(f"\n3. Running inference...")
    score_maps = get_score_map(args, models, support_imgs, test_imgs)

    # 4. Evaluation
    print(f"\n4. Calculating AUROC...")
    
    gt_pixel_list = []
    pred_pixel_list = []
    gt_image_list = []
    pred_image_list = []
    
    missing_masks = 0

    for i, name in enumerate(test_names):
        score_map = score_maps[i]
        
        # Try to find mask
        name_no_ext = os.path.splitext(name)[0]
        # Possible mask names: "image_mask.png", "image.png", "image_label.png"
        candidates = [
            os.path.join(args.mask_dir, name_no_ext + '_mask.png'),
            os.path.join(args.mask_dir, 'gt_' + name_no_ext + '.png'), # Support gt_ prefix
            os.path.join(args.mask_dir, 'gt_' + name),                 # Support gt_ prefix with original extension
            os.path.join(args.mask_dir, name_no_ext + '.png'),
            os.path.join(args.mask_dir, name),
            # Sometimes masks are inside a subdirectory with same name as image
            os.path.join(args.mask_dir, name_no_ext, name_no_ext + '_mask.png')
        ]
        
        mask = None
        for path in candidates:
            if os.path.exists(path):
                mask = load_mask(path, args.img_size)
                break
        
        if mask is None:
            raise FileNotFoundError(
                f"Missing ground-truth mask for {name}. Provide an explicit all-zero "
                "mask for normal images; missing annotations are not normal labels."
            )

        # Flatten for pixel AUROC
        gt_pixel_list.extend(mask.flatten())
        pred_pixel_list.extend(score_map.flatten())
        
        # Image level
        is_anomaly = 1 if np.sum(mask) > 0 else 0
        max_score = np.max(score_map)
        
        gt_image_list.append(is_anomaly)
        pred_image_list.append(max_score)

    # Calculate Metrics
    try:
        pixel_auroc = roc_auc_score(gt_pixel_list, pred_pixel_list)
    except Exception as e:
        print(f"Warning: Could not calc Pixel AUROC (maybe all masks are empty?): {e}")
        pixel_auroc = float("nan")

    try:
        image_auroc = roc_auc_score(gt_image_list, pred_image_list)
    except Exception as e:
        # This happens if there is only one class (e.g. all normal or all anomaly)
        print(f"Warning: Could not calc Image AUROC (Need both Normal and Anomaly sample in GT): {e}")
        image_auroc = float("nan")

    print("\n" + "="*40)
    print("RESULTS")
    print("="*40)
    print(f"Pixel-Level AUROC: {pixel_auroc:.4f}")
    print(f"Image-Level AUROC: {image_auroc:.4f}")
    if missing_masks > 0:
        print(f"(Warning: {missing_masks} images had no found mask file and were treated as Normal)")
    print("="*40)

    # Save to file
    with open(os.path.join(args.output_dir, "auroc_results.txt"), "w") as f:
        f.write(f"Model: {args.obj}\n")
        f.write(f"Pixel-Level AUROC: {pixel_auroc:.4f}\n")
        f.write(f"Image-Level AUROC: {image_auroc:.4f}\n")
        f.write(f"Missing Masks: {missing_masks}\n")

if __name__ == '__main__':
    main()

