import os
import argparse
import numpy as np
from cellpose import io, transforms


def get_image_mask_pairs(directory, img_filter=None):
    """
    Find pairs of <image_name>.tif and <image_name>_cp_masks.tif in directory.
    
    Args:
        directory: Path to directory containing image/mask pairs
        img_filter: Optional string to filter image names
    
    Returns:
        List of tuples: (image_path, mask_path)
    """
    pairs = []
    files = os.listdir(directory)
    
    # Find all image files
    for fname in files:
        if fname.endswith('.tif') or fname.endswith('.tiff'):
            # Skip mask files
            if '_cp_masks' in fname:
                continue
            
            # Apply filter if provided
            if img_filter and img_filter not in fname:
                continue
            
            # Check if corresponding mask exists
            base_name = fname.replace('.tif', '').replace('.tiff', '')
            mask_name = f"{base_name}_cp_masks.tif"
            mask_path = os.path.join(directory, mask_name)
            
            if os.path.exists(mask_path):
                image_path = os.path.join(directory, fname)
                pairs.append((image_path, mask_path))
    
    return pairs


def extract_planes(image_3d, mask_3d, nimg_per_view=10, crop_size=512, 
                   anisotropy=1.0, sharpen_radius=0.0, tile_norm=0):
    """
    Extract random XY, XZ, YZ planes from 3D image/mask pairs.
    Ensures same permutation is applied to both image and mask.
    
    Args:
        image_3d: 3D image array (Z, Y, X) or (Z, Y, X, C)
        mask_3d: 3D mask array (Z, Y, X)
        nimg_per_view: Number of random crops per view
        crop_size: Size of random crops
        anisotropy: Anisotropy ratio for Z-axis scaling
        sharpen_radius: Sharpening radius for preprocessing
        tile_norm: Tile normalization block size
    
    Returns:
        List of tuples: (plane_name, processed_image, processed_mask)
    """
    results = []
    
    # Define permutations: (view_name, permutation_axes, should_scale_z)
    permutations = [
        ("XY", (0, 1, 2), False),      # (Z, Y, X) - no scaling needed
        ("ZY", (2, 0, 1), True),       # (X, Z, Y) - apply anisotropy to Z
        ("ZX", (1, 0, 2), True),       # (Y, Z, X) - apply anisotropy to Z
    ]
    
    for view_name, perm, should_scale_z in permutations:
        # Transpose both image and mask with same permutation
        if image_3d.ndim == 4:  # Has channel dimension
            img_perm = image_3d.transpose(perm + (3,))  # Add channel at end
            mask_perm = mask_3d.transpose(perm)
        else:
            img_perm = image_3d.transpose(perm)
            mask_perm = mask_3d.transpose(perm)
        
        Ly, Lx = img_perm.shape[1:3]
        
        # Apply anisotropy scaling if needed (for Z-axis views)
        if should_scale_z and anisotropy > 1.0:
            new_Ly = int(anisotropy * Ly)
            img_perm = transforms.resize_image(img_perm, Ly=new_Ly, Lx=Lx)
            mask_perm = transforms.resize_image(mask_perm, Ly=new_Ly, Lx=Lx)
            Ly = new_Ly
        
        # Randomly select slices from this view
        nz = img_perm.shape[0]
        slice_indices = np.random.choice(nz, size=min(nimg_per_view, nz), replace=False)
        
        for i, z_idx in enumerate(slice_indices):
            img_slice = img_perm[z_idx]
            mask_slice = mask_perm[z_idx]
            
            # Apply preprocessing to image
            if tile_norm:
                img_slice = transforms.normalize99_tile(img_slice, blocksize=tile_norm)
            
            if sharpen_radius:
                img_slice = transforms.smooth_sharpen_img(img_slice, 
                                                          sharpen_radius=sharpen_radius)
            
            # Random crop
            ly = 0 if Ly - crop_size <= 0 else np.random.randint(0, Ly - crop_size)
            lx = 0 if Lx - crop_size <= 0 else np.random.randint(0, Lx - crop_size)
            
            img_crop = img_slice[ly:ly + crop_size, lx:lx + crop_size].squeeze()
            mask_crop = mask_slice[ly:ly + crop_size, lx:lx + crop_size].squeeze()
            
            results.append({
                'view': view_name,
                'slice_idx': z_idx,
                'image': img_crop,
                'mask': mask_crop,
                'basename': f"{view_name}_{i}"
            })
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Extract random XY/XZ/YZ planes from 3D image/mask pairs, '
                    'keeping same view for corresponding image and mask pairs.'
    )
    
    input_args = parser.add_argument_group("input arguments")
    input_args.add_argument('--dir', required=True, type=str,
                            help='directory containing image and mask pairs')
    input_args.add_argument('--img_filter', default='', type=str,
                            help='optional filter string for image names')
    input_args.add_argument('--output_dir', default=None, type=str,
                            help='output directory (default: <input_dir>/train/)')
    input_args.add_argument('--channel_axis', default=-1, type=int,
                            help='axis of image which corresponds to image channels. Default: %(default)s')
    input_args.add_argument('--z_axis', default=0, type=int,
                            help='axis of image which corresponds to Z dimension. Default: %(default)s')
    
    # Processing arguments
    proc_args = parser.add_argument_group("processing arguments")
    proc_args.add_argument('--nimg_per_view', required=False, default=10, type=int,
                           help='number of random slices per view (XY/XZ/YZ). Default: %(default)s')
    proc_args.add_argument('--crop_size', required=False, default=512, type=int,
                           help='size of random crops. Default: %(default)s')
    proc_args.add_argument('--anisotropy', required=False, default=1.0, type=float,
                           help='anisotropy ratio for Z-axis scaling in XZ/YZ views. Default: %(default)s')
    proc_args.add_argument('--sharpen_radius', required=False, default=0.0, type=float,
                           help='high-pass filtering radius. Default: %(default)s')
    proc_args.add_argument('--tile_norm', required=False, default=0, type=int,
                           help='tile normalization block size. Default: %(default)s')
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.isdir(args.dir):
        raise ValueError(f"ERROR: directory not found at {args.dir}")
    
    # Set output directory
    output_dir = args.output_dir if args.output_dir else os.path.join(args.dir, 'train')
    os.makedirs(output_dir, exist_ok=True)
    
    # Find image/mask pairs
    pairs = get_image_mask_pairs(args.dir, img_filter=args.img_filter if args.img_filter else None)
    
    if not pairs:
        print(f"WARNING: no image/mask pairs found in {args.dir}")
        return
    
    print(f"Found {len(pairs)} image/mask pairs")
    
    # Set random seed for reproducibility
    np.random.seed(0)
    
    # Process each pair
    total_saved = 0
    for image_path, mask_path in pairs:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        print(f"Processing: {base_name}")
        
        try:
            # Load 3D image and mask
            img_3d = io.imread_3D(image_path)
            mask_3d = io.imread_3D(mask_path)
            
            # Convert image to standard format (Z, Y, X, C)
            img_3d = transforms.convert_image(img_3d, channel_axis=args.channel_axis, 
                                               z_axis=args.z_axis, do_3D=True)
            mask_3d = transforms.convert_image(mask_3d, channel_axis=args.channel_axis,
                                               z_axis=args.z_axis, do_3D=True)
            
            # Ensure mask is single channel
            if mask_3d.ndim == 4 and mask_3d.shape[3] > 1:
                mask_3d = mask_3d[..., 0]
            if mask_3d.ndim == 4:
                mask_3d = mask_3d.squeeze()
            
            # Extract planes with matching permutations
            planes = extract_planes(
                img_3d, mask_3d,
                nimg_per_view=args.nimg_per_view,
                crop_size=args.crop_size,
                anisotropy=args.anisotropy,
                sharpen_radius=args.sharpen_radius,
                tile_norm=args.tile_norm
            )
            
            # Save extracted planes
            for plane_data in planes:
                basename = plane_data['basename']
                img = plane_data['image']
                mask = plane_data['mask']
                
                img_filename = f"{base_name}_{basename}.tif"
                mask_filename = f"{base_name}_{basename}_masks.tif"
                
                img_path = os.path.join(output_dir, img_filename)
                mask_path = os.path.join(output_dir, mask_filename)
                
                io.imsave(img_path, img)
                io.imsave(mask_path, mask.astype(np.uint16))
                
                total_saved += 1
            
            print(f"  Saved {len(planes)} image/mask pairs")
            
        except Exception as e:
            print(f"  ERROR processing {base_name}: {e}")
            continue
    
    print(f"\nTotal pairs saved: {total_saved}")
    print(f"Output saved to: {output_dir}")


if __name__ == '__main__':
    main()
