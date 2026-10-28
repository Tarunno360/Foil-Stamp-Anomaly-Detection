import os
import argparse
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from skimage import exposure
from models.siamese import Encoder, Predictor
from models.stn import stn_net
from losses.norm_loss import CosLoss
from utils.funcs import embedding_concat, mahalanobis_torch, rot_img, translation_img, hflip_img, rot90_img, grey_img
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter
from collections import OrderedDict
import warnings
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

def load_images_from_folder(folder_path, img_size=224):
    """Load all images from a folder"""
    images = []
    image_names = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(valid_extensions):
            img_path = os.path.join(folder_path, filename)
            img_tensor = load_image(img_path, img_size)
            images.append(img_tensor)
            image_names.append(filename)

    if len(images) == 0:
        raise ValueError(f"No valid images found in {folder_path}")

    return torch.stack(images), image_names

def augment_support_images(support_img):
    """Apply augmentation to support images"""
    augment_support_img = support_img

    # rotate img with small angle
    for angle in [-np.pi/4, -3*np.pi/16, -np.pi/8, -np.pi/16, np.pi/16, np.pi/8, 3*np.pi/16, np.pi/4]:
        rotate_img_tensor = rot_img(support_img, angle)
        augment_support_img = torch.cat([augment_support_img, rotate_img_tensor], dim=0)

    # translate img
    for a, b in [(0.2,0.2), (-0.2,0.2), (-0.2,-0.2), (0.2,-0.2), (0.1,0.1), (-0.1,0.1), (-0.1,-0.1), (0.1,-0.1)]:
        trans_img = translation_img(support_img, a, b)
        augment_support_img = torch.cat([augment_support_img, trans_img], dim=0)

    # hflip img
    flipped_img = hflip_img(support_img)
    augment_support_img = torch.cat([augment_support_img, flipped_img], dim=0)

    # rgb to grey img
    greyed_img = grey_img(support_img)
    augment_support_img = torch.cat([augment_support_img, greyed_img], dim=0)

    # rotate img in 90 degree
    for angle in [1, 2, 3]:
        rotate90_img_tensor = rot90_img(support_img, angle)
        augment_support_img = torch.cat([augment_support_img, rotate90_img_tensor], dim=0)

    # shuffle augmented images
    augment_support_img = augment_support_img[torch.randperm(augment_support_img.size(0))]

    return augment_support_img

def test_custom_images(args, models, support_imgs, test_imgs):
    """Test custom images against support set"""
    STN = models[0]
    ENC = models[1]
    PRED = models[2]

    STN.eval()
    ENC.eval()
    PRED.eval()

    train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
    test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

    # Augment support images
    augment_support_img = augment_support_images(support_imgs)

    # Extract features from augmented support images
    with torch.no_grad():
        support_feat = STN(augment_support_img.to(device))
    support_feat = torch.mean(support_feat, dim=0, keepdim=True)

    train_outputs['layer1'].append(STN.stn1_output)
    train_outputs['layer2'].append(STN.stn2_output)
    train_outputs['layer3'].append(STN.stn3_output)

    for k, v in train_outputs.items():
        train_outputs[k] = torch.cat(v, 0)

    # Embedding concat
    embedding_vectors = train_outputs['layer1']
    for layer_name in ['layer2', 'layer3']:
        embedding_vectors = embedding_concat(embedding_vectors, train_outputs[layer_name], use_cuda)

    # Calculate multivariate Gaussian distribution
    B, C, H, W = embedding_vectors.size()
    embedding_vectors = embedding_vectors.view(B, C, H * W)
    mean = torch.mean(embedding_vectors, dim=0)
    cov = torch.zeros(C, C, H * W).to(device)
    I = torch.eye(C).to(device)
    for i in range(H * W):
        cov[:, :, i] = torch.cov(embedding_vectors[:, :, i].T) + 0.01 * I
    train_outputs = [mean, cov]

    # Test each query image
    score_map_list = []

    for idx in range(test_imgs.size(0)):
        query_img = test_imgs[idx:idx+1].to(device)

        # Model prediction
        query_feat = STN(query_img)
        z1 = ENC(query_feat)
        z2 = ENC(support_feat)
        p1 = PRED(z1)
        p2 = PRED(z2)

        loss = CosLoss(p1, z2, Mean=False)/2 + CosLoss(p2, z1, Mean=False)/2
        loss_reshape = F.interpolate(loss.unsqueeze(1), size=args.img_size, mode='bilinear',
                                     align_corners=False).squeeze(0)
        score_map_list.append(loss_reshape.cpu().detach().numpy())

        test_outputs['layer1'].append(STN.stn1_output)
        test_outputs['layer2'].append(STN.stn2_output)
        test_outputs['layer3'].append(STN.stn3_output)

    for k, v in test_outputs.items():
        test_outputs[k] = torch.cat(v, 0)

    # Embedding concat for test images
    embedding_vectors = test_outputs['layer1']
    for layer_name in ['layer2', 'layer3']:
        embedding_vectors = embedding_concat(embedding_vectors, test_outputs[layer_name], use_cuda)

    # Calculate distance matrix
    B, C, H, W = embedding_vectors.size()
    embedding_vectors = embedding_vectors.view(B, C, H * W)
    dist_list = []

    for i in range(H * W):
        mean_val = train_outputs[0][:, i]
        conv_inv = torch.linalg.inv(train_outputs[1][:, :, i])
        dist = [mahalanobis_torch(sample[:, i], mean_val, conv_inv) for sample in embedding_vectors]
        dist_list.append(dist)

    dist_list = torch.tensor(dist_list).transpose(1, 0).reshape(B, H, W)

    # Upsample
    score_map = F.interpolate(dist_list.unsqueeze(1), size=args.img_size, mode='bilinear',
                             align_corners=False).squeeze(1).cpu().numpy()

    # Apply gaussian smoothing on the score map
    for i in range(score_map.shape[0]):
        score_map[i] = gaussian_filter(score_map[i], sigma=4)

    return score_map, np.array(score_map_list)

def create_binary_mask(score_map_normalized, threshold):
    """
    Create a binary mask from normalized anomaly score map.

    Pixels with normalized score > threshold are considered anomalies (1), others 0.

    Args:
        score_map_normalized (np.ndarray): Normalized anomaly score map (2D array, 0-1)
        threshold (float): Threshold value (0-1)

    Returns:
        np.ndarray: Binary mask (0 for normal, 1 for anomaly)
    """
    # Create binary mask: 1 where normalized score > threshold, else 0
    binary_mask = (score_map_normalized > threshold).astype(np.uint8)

    return binary_mask

def visualize_anomaly(image_path, score_map, save_path, threshold=0.5, anomaly_score=0.0):
    """Visualize anomaly by overlaying heatmap on original image"""
    # Load original image
    original_img = Image.open(image_path).convert('RGB')
    original_img = np.array(original_img)

    # Resize score_map to match original image size if necessary
    if score_map.shape != original_img.shape[:2]:
        from scipy.ndimage import zoom
        zoom_factor = (original_img.shape[0] / score_map.shape[0], original_img.shape[1] / score_map.shape[1])
        score_map = zoom(score_map, zoom_factor, order=1)

    # If the anomaly score is below or equal to threshold (NORMAL), show no anomaly visualization
    if anomaly_score <= threshold:
        score_map_normalized = np.zeros_like(score_map)
    else:
        # Clip negative values to 0
        score_map = np.clip(score_map, 0, None)

        # Normalize score_map for visualization
        if score_map.max() > score_map.min():
            score_map_normalized = (score_map - score_map.min()) / (score_map.max() - score_map.min() + 1e-8)
        else:
            score_map_normalized = np.zeros_like(score_map)

    # Create heatmap
    heatmap = plt.cm.jet(score_map_normalized)[:, :, :3]  # Remove alpha channel

    # Overlay heatmap on original image
    overlay = 0.6 * original_img / 255.0 + 0.4 * heatmap
    overlay = np.clip(overlay, 0, 1)

    # Create binary mask
    # Use a high threshold (0.66) to capture only the "highest darker portion" (red/orange areas) of the heatmap
    mask_threshold = 0.66 
    binary_mask = create_binary_mask(score_map_normalized, mask_threshold)

    # Create figure with 4 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Original image
    ax1.imshow(original_img)
    ax1.set_title('Original Image')
    ax1.axis('off')

    # Heatmap
    im = ax2.imshow(score_map_normalized, cmap='jet', vmin=0, vmax=1)
    ax2.set_title('Anomaly Score Map')
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    # Overlay
    ax3.imshow(overlay)
    ax3.set_title('Anomaly Overlay')
    ax3.axis('off')

    # Binary Mask
    ax4.imshow(binary_mask, cmap='gray', vmin=0, vmax=1)
    ax4.set_title('Binary Anomaly Mask (White = Anomaly)')
    ax4.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='RegAD Foil-Stamp Inference')
    parser.add_argument('--obj', type=str, default='leather', help='Object category for loading the model')
    parser.add_argument('--shot', type=int, default=2, help='Shot count (used for loading model)')
    parser.add_argument('--stn_mode', type=str, default='rotation_scale',
                       help='STN mode used during training')
    parser.add_argument('--img_size', type=int, default=224, help='Image size')
    parser.add_argument('--support_dir', type=str, required=True,
                       help='Directory containing support (normal) images')
    parser.add_argument('--test_dir', type=str, required=True,
                       help='Directory containing test images')
    parser.add_argument('--threshold', type=float, required=True,
                       help='Raw anomaly-score threshold calibrated on separate validation data')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                       help='Custom path to model checkpoint (optional)')
    parser.add_argument('--output_dir', type=str, default='./anomaly_visualizations',
                       help='Directory to save visualization images')

    args = parser.parse_args()
    args.input_channel = 3

    print("="*80)
    print("Custom Image Anomaly Detection with Visualization (Fixed)")
    print("="*80)

    # Create output directory for visualizations
    os.makedirs(args.output_dir, exist_ok=True)

    # Load models
    print(f"\n1. Loading trained model for '{args.obj}' (shot={args.shot})...")
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
            print(f"Example: --checkpoint_path \"./save_checkpoints/2/leather/leather_2_rotation_scale_model.pt\"")
            return

    if not os.path.exists(CKPT_name):
        print(f"ERROR: Model checkpoint not found at {CKPT_name}")
        print("Please ensure you have trained the model or downloaded the pre-trained weights.")
        return

    model_CKPT = torch.load(CKPT_name, map_location=device)
    STN.load_state_dict(model_CKPT['STN'])
    ENC.load_state_dict(model_CKPT['ENC'])
    PRED.load_state_dict(model_CKPT['PRED'])
    models = [STN, ENC, PRED]
    print("✓ Model loaded successfully!")

    # Load support images
    print(f"\n2. Loading support (normal) images from: {args.support_dir}")
    support_imgs, support_names = load_images_from_folder(args.support_dir, args.img_size)
    print(f"✓ Loaded {len(support_names)} support images:")
    for name in support_names:
        print(f"   - {name}")

    # Load test images
    print(f"\n3. Loading test images from: {args.test_dir}")
    test_imgs, test_names = load_images_from_folder(args.test_dir, args.img_size)
    print(f"✓ Loaded {len(test_names)} test images:")
    for name in test_names:
        print(f"   - {name}")

    # Run anomaly detection
    print(f"\n4. Running anomaly detection...")
    print("   (This may take a moment...)")

    with torch.no_grad():
        score_maps, cos_scores = test_custom_images(args, models, support_imgs, test_imgs)

    # Calculate anomaly scores
    scores = score_maps.reshape(score_maps.shape[0], -1).max(axis=1)

    # Use fixed raw scores; predictions must not depend on test-batch composition.
    normalized_scores = scores

    # Display results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\nAnomaly Threshold: {args.threshold:.2f}")
    print("-"*80)
    print(f"{'Image Name':<30} {'Anomaly Score':<20} {'Prediction':<15}")
    print("-"*80)

    anomaly_count = 0
    for i, (name, score) in enumerate(zip(test_names, normalized_scores)):
        is_anomaly = score > args.threshold
        prediction = "ANOMALY" if is_anomaly else "NORMAL"
        anomaly_count += int(is_anomaly)

        # Color coding for terminal
        if is_anomaly:
            print(f"{name:<30} {score:<20.4f} ❌ {prediction:<15}")
        else:
            print(f"{name:<30} {score:<20.4f} ✓ {prediction:<15}")

    print("-"*80)
    print(f"\nSummary:")
    print(f"  Total test images: {len(test_names)}")
    print(f"  Detected anomalies: {anomaly_count}")
    print(f"  Normal images: {len(test_names) - anomaly_count}")
    print("="*80)

    # Generate visualizations
    print(f"\n5. Generating anomaly visualizations...")
    print(f"   Saving to: {args.output_dir}")

    for i, (name, score_map, score) in enumerate(zip(test_names, score_maps, normalized_scores)):
        test_img_path = os.path.join(args.test_dir, name)
        vis_save_path = os.path.join(args.output_dir, f"{os.path.splitext(name)[0]}_anomaly_vis.png")

        try:
            visualize_anomaly(test_img_path, score_map, vis_save_path, args.threshold, score)
            print(f"   ✓ Saved visualization for {name}")
        except Exception as e:
            print(f"   ✗ Failed to save visualization for {name}: {str(e)}")

    print(f"\n✓ All visualizations saved to: {args.output_dir}")

    # Save detailed results
    output_file = os.path.join(args.output_dir, "predictions.txt")
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("Anomaly Detection Results\n")
        f.write("="*80 + "\n\n")
        f.write(f"Support Directory: {args.support_dir}\n")
        f.write(f"Test Directory: {args.test_dir}\n")
        f.write(f"Model: {args.obj} (shot={args.shot})\n")
        f.write(f"Threshold: {args.threshold}\n\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Image Name':<30} {'Anomaly Score':<20} {'Prediction':<15}\n")
        f.write("-"*80 + "\n")

        for name, score in zip(test_names, normalized_scores):
            is_anomaly = score > args.threshold
            prediction = "ANOMALY" if is_anomaly else "NORMAL"
            f.write(f"{name:<30} {score:<20.4f} {prediction:<15}\n")

        f.write("-"*80 + "\n")
        f.write(f"\nTotal test images: {len(test_names)}\n")
        f.write(f"Detected anomalies: {anomaly_count}\n")
        f.write(f"Normal images: {len(test_names) - anomaly_count}\n")

    print(f"\n✓ Detailed results saved to: {output_file}\n")

if __name__ == '__main__':
    main()

