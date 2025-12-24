class TiledImageDirDataset(Dataset):
    """
    Dataset for a single image directory that has a precomputed tile_index.npy.

    Directory structure example:
        root_dir/
            img_0001.tif
            img_0002.png
            ...
            tile_index.npy
            tile_index_mask_info.npy (optional, if masks_enabled=True)

    tile_index.npy format (per-directory):
        Shape: (N, 6), (N, 7), or (N, 5)
        Columns:
          - if 7: (ds_idx, sample_idx, y, x, H, W, mask_sample_idx)  # with masks
          - if 6: (ds_idx, sample_idx, y, x, H, W)  # ds_idx is ignored
          - if 5: (sample_idx, y, x, H, W)

    tile_index_mask_info.npy format (optional):
        Dictionary mapping sample_idx -> mask_path (string)

    Arguments:
        root_dir (str or Path): directory containing images + tile_index.npy
        tile_size (int): tile edge size (B) — tiles are B x B
        dtype (torch.dtype): output tensor dtype
        tile_index_filename (str): name of the .npy file with tile index
        recursive (bool): whether to search for images recursively
        masks_enabled (bool): whether to load masks from mask files
    """

    def __init__(
        self,
        root_dir,
        tile_size=256,
        dtype=torch.float16,
        tile_index_filename="tile_index.npy",
        recursive=False,
        masks_enabled=False,
    ):
        self.root_dir = Path(root_dir)
        self.tile_size = tile_size
        self.dtype = dtype
        self.recursive = recursive
        self.masks_enabled = masks_enabled

        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"Directory not found: {self.root_dir}")

        # 1) Load tile_index for this directory
        tile_index_path = self.root_dir / tile_index_filename
        if not tile_index_path.is_file():
            raise FileNotFoundError(f"tile_index file not found: {tile_index_path}")

        tile_index = np.load(tile_index_path)
        tile_index = np.asarray(tile_index, dtype=np.int64)

        if tile_index.ndim != 2 or tile_index.shape[1] not in (5, 6, 7):
            raise ValueError(
                f"tile_index at {tile_index_path} must have shape (N,5), (N,6), or (N,7), "
                f"got {tile_index.shape}"
            )

        # Check if masks are present in tile_index
        has_mask_column = tile_index.shape[1] == 7

        # Drop ds_idx column if present, extract mask_sample_idx if present
        if tile_index.shape[1] == 7:
            self.mask_sample_indices = tile_index[:, 6]  # Extract mask indices
            tile_index = tile_index[:, 1:6]  # (sample_idx, y, x, H, W)
        elif tile_index.shape[1] == 6:
            tile_index = tile_index[:, 1:]  # (sample_idx, y, x, H, W)
            self.mask_sample_indices = None
        else:  # shape[1] == 5
            self.mask_sample_indices = None

        # Store per-tile info: (sample_idx, y, x, H, W)
        self.tile_index = tile_index

        # Load mask info if masks are enabled
        self.mask_info = None
        self.mask_paths = {}
        if self.masks_enabled:
            if not has_mask_column:
                logger.warning(
                    f"masks_enabled=True but tile_index does not have mask column. "
                    f"Masks will not be loaded."
                )
                self.masks_enabled = False
            else:
                # Load mask_info file
                mask_info_path = self.root_dir / (
                    Path(tile_index_filename).stem + "_mask_info.npy"
                )
                if mask_info_path.is_file():
                    self.mask_info = np.load(mask_info_path, allow_pickle=True).item()
                    logger.info(f"Loaded mask info with {len(self.mask_info)} entries")
                else:
                    logger.warning(
                        f"masks_enabled=True but mask_info file not found: {mask_info_path}. "
                        f"Masks will not be loaded."
                    )
                    self.masks_enabled = False

        # 2) Reconstruct the *same* file ordering as used when tile_index was computed
        # In compute_tile_index_dir.py we used: sorted(files) with optional recursion
        # Filter for common image extensions only (exclude .npy and other non-image files)
        # Also exclude flow files (_flows.*) and mask files (_masks.*)
        image_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}
        if self.recursive:
            self.image_paths = sorted(
                p
                for p in self.root_dir.rglob("*")
                if p.is_file() 
                and p.suffix.lower() in image_extensions
                and not p.stem.endswith('_flows')
                and not p.stem.endswith('_masks')
            )
        else:
            self.image_paths = sorted(
                p
                for p in self.root_dir.iterdir()
                if p.is_file() 
                and p.suffix.lower() in image_extensions
                and not p.stem.endswith('_flows')
                and not p.stem.endswith('_masks')
            )

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No image files found in directory: {self.root_dir}")

        mask_status = ""
        if self.masks_enabled and self.mask_info:
            num_with_masks = sum(
                1 for idx in self.mask_info.values() if idx is not None
            )
            mask_status = f", {num_with_masks} masks"

        print(
            f"TiledImageDirDataset: {len(self.tile_index)} tiles from "
            f"{len(self.image_paths)} files{mask_status} in {self.root_dir}"
        )

    # ---------- helpers ----------

    def _load_image(self, sample_idx: int):
        """
        Load the original image corresponding to sample_idx using PIL,
        convert to HWC numpy array with channels last.
        """
        img_path = self.image_paths[sample_idx]

        try:
            with Image.open(img_path) as img:
                # Force load the entire image to catch truncation errors early
                img.load()
                img = img.convert("RGB") if img.mode not in ("L", "RGB") else img.copy()
                arr = np.array(img)
        except (OSError, IOError) as e:
            # Log the error and return a black image as fallback
            logger.warning(
                f"Failed to load image {img_path}: {e}. Using black image as fallback."
            )
            # Get expected dimensions from tile_index if available
            if hasattr(self, "tile_index") and sample_idx < len(self.tile_index):
                # Use first occurrence of this sample_idx
                mask = self.tile_index[:, 0] == sample_idx
                if mask.any():
                    idx = np.where(mask)[0][0]
                    H, W = int(self.tile_index[idx, 3]), int(self.tile_index[idx, 4])
                    arr = np.zeros((H, W, 3), dtype=np.uint8)
                else:
                    # Fallback to a default size
                    arr = np.zeros((1024, 1024, 3), dtype=np.uint8)
            else:
                # Fallback to a default size
                arr = np.zeros((1024, 1024, 3), dtype=np.uint8)

        # (H, W) -> (H, W, 1)
        if arr.ndim == 2:
            arr = arr[..., None]
        return arr  # HWC

    def _load_mask(self, mask_sample_idx: int):
        """
        Load the mask corresponding to mask_sample_idx using PIL,
        convert to HW numpy array.
        Returns None if mask_sample_idx is -1 or mask path not found.
        """
        if mask_sample_idx < 0:
            return None

        if self.mask_info is None or mask_sample_idx not in self.mask_info:
            return None

        mask_info_entry = self.mask_info[mask_sample_idx]
        
        # Handle both old format (string path) and new format (dict with mask_path/flow_path)
        if isinstance(mask_info_entry, str):
            mask_path = Path(mask_info_entry)
        else:
            mask_path = Path(mask_info_entry['mask_path'])

        # Make path absolute if it's relative
        # if not mask_path.is_absolute():
        #     mask_path = self.root_dir / mask_path

        try:
            with Image.open(mask_path) as mask_img:
                mask_img.load()
                # Convert to grayscale/single channel
                if mask_img.mode not in ("L", "I"):
                    mask_img = mask_img.convert("L")
                mask_arr = np.array(mask_img)
        except (OSError, IOError) as e:
            logger.warning(f"Failed to load mask {mask_path}: {e}. Returning None.")
            return None

        return mask_arr  # (H, W)
    
    def _load_flow(self, mask_sample_idx: int):
        """
        Load precomputed flow corresponding to mask_sample_idx.
        Returns None if flow not available or mask_sample_idx is -1.
        
        Returns:
            np.ndarray or None: Flow array of shape (4, H, W) where:
                - flow[0] = labels (masks)
                - flow[1] = cell distance transform
                - flow[2] = Y flow
                - flow[3] = X flow
        """
        if mask_sample_idx < 0:
            return None
            
        if self.mask_info is None or mask_sample_idx not in self.mask_info:
            return None
        
        mask_info_entry = self.mask_info[mask_sample_idx]
        
        # Only new format (dict) supports flow_path
        if not isinstance(mask_info_entry, dict):
            return None
            
        flow_path_str = mask_info_entry.get('flow_path')
        if flow_path_str is None:
            return None
        
        flow_path = Path(flow_path_str)
        
        try:
            # Load flow using tifffile (flows are saved as multi-channel TIFF)
            flow_arr = tifffile.imread(str(flow_path))
            
            # Validate shape: should be (4, H, W)
            if flow_arr.ndim != 3 or flow_arr.shape[0] != 4:
                logger.warning(
                    f"Invalid flow shape {flow_arr.shape} for {flow_path}. Expected (4, H, W)."
                )
                return None
                
            return flow_arr.astype(np.float32)
        except Exception as e:
            logger.warning(f"Failed to load flow {flow_path}: {e}. Returning None.")
            return None

    def _image_array_to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        """
        HWC numpy -> normalized CHW torch tensor in self.dtype
        """
        arr = arr.astype(np.float32) / 255.0

        tensor = torch.from_numpy(arr).permute(2, 0, 1).to(self.dtype)
        # Pad the channels dimension to be of size 3 with zeros if necessary
        if tensor.shape[0] < 3:
            padding = torch.zeros(
                3 - tensor.shape[0],
                tensor.shape[1],
                tensor.shape[2],
                dtype=tensor.dtype,
                device=tensor.device,
            )
            tensor = torch.cat([tensor, padding], dim=0)
        return tensor

    # ---------- Dataset API ----------

    def __len__(self):
        return self.tile_index.shape[0]

    def __getitem__(self, idx):
        """
        Returns:
            dict with keys "pixel_values" and "labels"
            - pixel_values: tensor(C, tile_size, tile_size)
            - labels: tensor(1, tile_size, tile_size) - mask or dummy zeros
        """
        sample_idx, y, x, H, W = self.tile_index[idx]
        tile_size = self.tile_size

        # Load original image
        arr = self._load_image(sample_idx)  # (H, W, C) ideally

        # Use stored H, W only as sanity if you want:
        # assert arr.shape[0] == H and arr.shape[1] == W

        # Extract image tile
        y_end = min(y + tile_size, arr.shape[0])
        x_end = min(x + tile_size, arr.shape[1])
        tile = arr[y:y_end, x:x_end, :]

        # Pad image tile if necessary
        h, w = tile.shape[:2]
        if h < tile_size or w < tile_size:
            padded = np.zeros((tile_size, tile_size, tile.shape[2]), dtype=tile.dtype)
            padded[:h, :w, :] = tile
            tile = padded

        img_tensor = self._image_array_to_tensor(tile)

        # Load mask/flow if enabled
        if self.masks_enabled and self.mask_sample_indices is not None:
            mask_sample_idx = int(self.mask_sample_indices[idx])
            
            # Try to load precomputed flow first
            flow_arr = self._load_flow(mask_sample_idx)
            
            if flow_arr is not None:
                # Use precomputed flow (4, H, W): [labels, cellprob, dY, dX]
                # Extract flow tile
                flow_y_end = min(y + tile_size, flow_arr.shape[1])
                flow_x_end = min(x + tile_size, flow_arr.shape[2])
                flow_tile = flow_arr[:, y:flow_y_end, x:flow_x_end]
                
                # Pad flow tile if necessary
                if flow_tile.shape[1] < tile_size or flow_tile.shape[2] < tile_size:
                    padded_flow = np.zeros(
                        (4, tile_size, tile_size), dtype=flow_tile.dtype
                    )
                    padded_flow[:, :flow_tile.shape[1], :flow_tile.shape[2]] = flow_tile
                    flow_tile = padded_flow
                
                # Convert flow to tensor - shape (4, H, W)
                # We only need the mask (first channel) for labels
                mask_tensor = (
                    torch.from_numpy(flow_tile[0:1]).to(self.dtype)
                )  # (1, H, W)
                
                # Store full flow for training (optional - can be used by trainer)
                flow_tensor = torch.from_numpy(flow_tile).to(self.dtype)  # (4, H, W)
                
                return {
                    "pixel_values": img_tensor, 
                    "labels": mask_tensor,
                    "flows": flow_tensor  # Precomputed flows
                }
            else:
                # Fall back to loading just the mask
                mask_arr = self._load_mask(mask_sample_idx)

                if mask_arr is not None:
                    # Extract mask tile
                    mask_y_end = min(y + tile_size, mask_arr.shape[0])
                    mask_x_end = min(x + tile_size, mask_arr.shape[1])
                    mask_tile = mask_arr[y:mask_y_end, x:mask_x_end]

                    # Pad mask tile if necessary
                    if mask_tile.shape[0] < tile_size or mask_tile.shape[1] < tile_size:
                        padded_mask = np.zeros(
                            (tile_size, tile_size), dtype=mask_tile.dtype
                        )
                        padded_mask[: mask_tile.shape[0], : mask_tile.shape[1]] = mask_tile
                        mask_tile = padded_mask

                    # Convert mask to tensor
                    mask_tensor = (
                        torch.from_numpy(mask_tile).unsqueeze(0).to(self.dtype)
                    )  # (1, H, W)
                else:
                    # Fallback to dummy labels if mask loading failed
                    mask_tensor = torch.zeros((1, tile_size, tile_size), dtype=self.dtype)
        else:
            # Add dummy labels for compatibility with data collators expecting labels
            mask_tensor = torch.zeros((1, tile_size, tile_size), dtype=self.dtype)

        return {"pixel_values": img_tensor, "labels": mask_tensor}