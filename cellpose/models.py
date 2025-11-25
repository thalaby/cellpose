"""
Copyright © 2025 Howard Hughes Medical Institute, Authored by Carsen Stringer, Michael Rariden and Marius Pachitariu.
"""

import os, time
from pathlib import Path
import numpy as np
from tqdm import trange
import torch
from scipy.ndimage import gaussian_filter
import gc
import cv2
from mobile_sam import sam_model_registry
import logging

models_logger = logging.getLogger(__name__)

from . import transforms, dynamics, utils, plot
from .vit_sam import Transformer
from .core import assign_device, run_net, run_3D
import time

_CPSAM_MODEL_URL = "https://huggingface.co/mouseland/cellpose-sam/resolve/main/cpsam"
_MODEL_DIR_ENV = os.environ.get("CELLPOSE_LOCAL_MODELS_PATH")
_MODEL_DIR_DEFAULT = Path.home().joinpath(".cellpose", "models")
MODEL_DIR = Path(_MODEL_DIR_ENV) if _MODEL_DIR_ENV else _MODEL_DIR_DEFAULT

MODEL_NAMES = ["cpsam"]

MODEL_LIST_PATH = os.fspath(MODEL_DIR.joinpath("gui_models.txt"))

normalize_default = {
    "lowhigh": None,
    "percentile": None,
    "normalize": True,
    "norm3D": True,
    "sharpen_radius": 0,
    "smooth_radius": 0,
    "tile_norm_blocksize": 0,
    "tile_norm_smooth3D": 1,
    "invert": False
}


def model_path(model_type, model_index=0):
    return cache_CPSAM_model_path()


def cache_CPSAM_model_path():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cached_file = os.fspath(MODEL_DIR.joinpath('cpsam'))
    if not os.path.exists(cached_file):
        models_logger.info('Downloading: "{}" to {}\n'.format(_CPSAM_MODEL_URL, cached_file))
        utils.download_url_to_file(_CPSAM_MODEL_URL, cached_file, progress=True)
    return cached_file


def get_user_models():
    model_strings = []
    if os.path.exists(MODEL_LIST_PATH):
        with open(MODEL_LIST_PATH, "r") as textfile:
            lines = [line.rstrip() for line in textfile]
            if len(lines) > 0:
                model_strings.extend(lines)
    return model_strings


class CellposeModel():
    """
    Class representing a Cellpose model.

    Attributes:
        diam_mean (float): Mean "diameter" value for the model.
        builtin (bool): Whether the model is a built-in model or not.
        device (torch device): Device used for model running / training.
        nclasses (int): Number of classes in the model.
        nbase (list): List of base values for the model.
        net (CPnet): Cellpose network.
        pretrained_model (str): Path to pretrained cellpose model.
        pretrained_model_ortho (str): Path or model_name for pretrained cellpose model for ortho views in 3D.
        backbone (str): Type of network ("default" is the standard res-unet, "transformer" for the segformer).

    Methods:
        __init__(self, gpu=False, pretrained_model=False, model_type=None, diam_mean=30., device=None):
            Initialize the CellposeModel.
        
        eval(self, x, batch_size=8, resample=True, channels=None, channel_axis=None, z_axis=None, normalize=True, invert=False, rescale=None, diameter=None, flow_threshold=0.4, cellprob_threshold=0.0, do_3D=False, anisotropy=None, stitch_threshold=0.0, min_size=15, niter=None, augment=False, tile_overlap=0.1, bsize=256, interp=True, compute_masks=True, progress=None):
            Segment list of images x, or 4D array - Z x C x Y x X.

    """

    def __init__(self, gpu=False, pretrained_model="cpsam", model_type=None,
                 diam_mean=None, device=None, nchan=None, use_bfloat16=True):
        """
        Initialize the CellposeModel.

        Parameters:
            gpu (bool, optional): Whether or not to save model to GPU, will check if GPU available.
            pretrained_model (str or list of strings, optional): Full path to pretrained cellpose model(s), if None or False, no model loaded.
            model_type (str, optional): Any model that is available in the GUI, use name in GUI e.g. "livecell" (can be user-trained or model zoo).
            diam_mean (float, optional): Mean "diameter", 30. is built-in value for "cyto" model; 17. is built-in value for "nuclei" model; if saved in custom model file (cellpose>=2.0) then it will be loaded automatically and overwrite this value.
            device (torch device, optional): Device used for model running / training (torch.device("cuda") or torch.device("cpu")), overrides gpu input, recommended if you want to use a specific GPU (e.g. torch.device("cuda:1")).
            use_bfloat16 (bool, optional): Use 16bit float precision instead of 32bit for model weights. Default to 16bit (True).
        """
        if diam_mean is not None:
            models_logger.warning(
                "diam_mean argument are not used in v4.0.1+. Ignoring this argument..."
            )
        if model_type is not None:
            models_logger.warning(
                "model_type argument is not used in v4.0.1+. Ignoring this argument..."
            )
        if nchan is not None:
            models_logger.warning("nchan argument is deprecated in v4.0.1+. Ignoring this argument")

        ### assign model device
        self.device = assign_device(gpu=gpu)[0] if device is None else device
        if torch.cuda.is_available():
            device_gpu = self.device.type == "cuda"
        elif torch.backends.mps.is_available():
            device_gpu = self.device.type == "mps"
        else:
            device_gpu = False
        self.gpu = device_gpu

        if pretrained_model is None:
            raise ValueError("Must specify a pretrained model, training from scratch is not implemented")
        
        ### create neural network
        if pretrained_model and not os.path.exists(pretrained_model):
            # check if pretrained model is in the models directory
            model_strings = get_user_models()
            all_models = MODEL_NAMES.copy()
            all_models.extend(model_strings)
            if pretrained_model in all_models:
                pretrained_model = os.path.join(MODEL_DIR, pretrained_model)
            else:
                pretrained_model = os.path.join(MODEL_DIR, "cpsam")
                models_logger.warning(
                    f"pretrained model {pretrained_model} not found, using default model"
                )

        self.pretrained_model = pretrained_model
        dtype = torch.bfloat16 if use_bfloat16 else torch.float32
        self.net = Transformer(dtype=dtype).to(self.device)

        if os.path.exists(self.pretrained_model):
            models_logger.info(f">>>> loading model {self.pretrained_model}")
            # self.net.load_model(self.pretrained_model, device=self.device)
        else:
            if os.path.split(self.pretrained_model)[-1] != 'cpsam':
                raise FileNotFoundError('model file not recognized')
            cache_CPSAM_model_path()
            # self.net.load_model(self.pretrained_model, device=self.device)
        
        
    def eval(self, x, batch_size=8, resample=True, channels=None, channel_axis=None,
             z_axis=None, normalize=True, invert=False, rescale=None, diameter=None,
             flow_threshold=0.4, cellprob_threshold=0.0, do_3D=False, anisotropy=None,
             flow3D_smooth=0, stitch_threshold=0.0, 
             min_size=15, max_size_fraction=0.4, niter=None, 
             augment=False, tile_overlap=0.1, bsize=256, 
             compute_masks=True, progress=None):
        """ segment list of images x, or 4D array - Z x 3 x Y x X
        ... (docstring unchanged) ...
        """

        if rescale is not None:
            models_logger.warning("rescaling deprecated in v4.0.1+") 
        if channels is not None:
            models_logger.warning("channels deprecated in v4.0.1+. If data contain more than 3 channels, only the first 3 channels will be used")

        if isinstance(x, list) or x.squeeze().ndim == 5:
            self.timing = []
            masks, styles, flows = [], [], []
            tqdm_out = utils.TqdmToLogger(models_logger, level=logging.INFO)
            nimg = len(x)
            iterator = trange(nimg, file=tqdm_out,
                              mininterval=30) if nimg > 1 else range(nimg)
            for i in iterator:
                tic = time.time()
                maski, flowi, stylei = self.eval(
                    x[i], 
                    batch_size=batch_size,
                    channel_axis=channel_axis, 
                    z_axis=z_axis,
                    normalize=normalize, 
                    invert=invert,
                    diameter=diameter[i] if isinstance(diameter, list) or
                        isinstance(diameter, np.ndarray) else diameter, 
                    do_3D=do_3D,
                    anisotropy=anisotropy, 
                    augment=augment, 
                    tile_overlap=tile_overlap, 
                    bsize=bsize, 
                    resample=resample,
                    flow_threshold=flow_threshold,
                    cellprob_threshold=cellprob_threshold, 
                    compute_masks=compute_masks,
                    min_size=min_size, 
                    max_size_fraction=max_size_fraction, 
                    stitch_threshold=stitch_threshold, 
                    flow3D_smooth=flow3D_smooth,
                    progress=progress, 
                    niter=niter)
                masks.append(maski)
                flows.append(flowi)
                styles.append(stylei)
                self.timing.append(time.time() - tic)
            return masks, flows, styles

        # --- Profiling starts here ---
        timings = {}
        t_start = time.time()

        # reshape image
        t0 = time.time()
        x = transforms.convert_image(x, channel_axis=channel_axis,
                                        z_axis=z_axis, 
                                        do_3D=(do_3D or stitch_threshold > 0))
        timings['convert_image'] = time.time() - t0

        # Add batch dimension if not present
        t0 = time.time()
        if x.ndim < 4:
            x = x[np.newaxis, ...]
        nimg = x.shape[0]
        timings['add_batch_dim'] = time.time() - t0
        
        image_scaling = None
        Ly_0 = x.shape[1]
        Lx_0 = x.shape[2]
        Lz_0 = None
        if do_3D or stitch_threshold > 0:
            Lz_0 = x.shape[0]
        t0 = time.time()
        if diameter is not None:
            image_scaling = 30. / diameter
            x = transforms.resize_image(x,
                                        Ly=int(x.shape[1] * image_scaling),
                                        Lx=int(x.shape[2] * image_scaling))
        timings['resize_image'] = time.time() - t0

        # normalize image
        t0 = time.time()
        normalize_params = normalize_default
        if isinstance(normalize, dict):
            normalize_params = {**normalize_params, **normalize}
        elif not isinstance(normalize, bool):
            raise ValueError("normalize parameter must be a bool or a dict")
        else:
            normalize_params["normalize"] = normalize
            normalize_params["invert"] = invert

        # pre-normalize if 3D stack for stitching or do_3D
        do_normalization = True if normalize_params["normalize"] else False
        if nimg > 1 and do_normalization and (stitch_threshold or do_3D):
            normalize_params["norm3D"] = True if do_3D else normalize_params["norm3D"]
            x = transforms.normalize_img(x, **normalize_params)
            do_normalization = False # do not normalize again
        else:
            if normalize_params["norm3D"] and nimg > 1 and do_normalization:
                models_logger.warning(
                    "normalize_params['norm3D'] is True but do_3D is False and stitch_threshold=0, so setting to False"
                )
                normalize_params["norm3D"] = False
        if do_normalization:
            x = transforms.normalize_img(x, **normalize_params)
        timings['normalize_img'] = time.time() - t0

        # ajust the anisotropy when diameter is specified and images are resized:
        t0 = time.time()
        if isinstance(anisotropy, (float, int)) and image_scaling:
            anisotropy = image_scaling * anisotropy
        timings['anisotropy_adjust'] = time.time() - t0

        # Run network
        t0 = time.time()
        dP, cellprob, styles = self._run_net(
            x, 
            augment=augment, 
            batch_size=batch_size, 
            tile_overlap=tile_overlap, 
            bsize=bsize,
            do_3D=do_3D, 
            anisotropy=anisotropy)
        timings['run_net'] = time.time() - t0

        # 3D smoothing
        t0 = time.time()
        if do_3D:    
            if flow3D_smooth > 0:
                models_logger.info(f"smoothing flows with sigma={flow3D_smooth}")
                dP = gaussian_filter(dP, (0, flow3D_smooth, flow3D_smooth, flow3D_smooth))
            torch.cuda.empty_cache()
            gc.collect()
        timings['3D_smooth'] = time.time() - t0

        # Resample
        t0 = time.time()
        if resample:
            # upsample flows before computing them: 
            dP = self._resize_gradients(dP, to_y_size=Ly_0, to_x_size=Lx_0, to_z_size=Lz_0)
            cellprob = self._resize_cellprob(cellprob, to_x_size=Lx_0, to_y_size=Ly_0, to_z_size=Lz_0)
        timings['resample'] = time.time() - t0

        # Compute masks
        t0 = time.time()
        if compute_masks:
            # use user niter if specified, otherwise scale niter (200) with diameter
            niter_scale = 1 if image_scaling is None else image_scaling
            niter = int(200/niter_scale) if niter is None or niter == 0 else niter
            masks = self._compute_masks(x.shape, dP, cellprob, flow_threshold=flow_threshold,
                            cellprob_threshold=cellprob_threshold, min_size=min_size,
                        max_size_fraction=max_size_fraction, niter=niter,
                        stitch_threshold=stitch_threshold, do_3D=do_3D)
        else:
            masks = np.zeros(0) #pass back zeros if not compute_masks
        timings['compute_masks'] = time.time() - t0
        
        t0 = time.time()
        masks, dP, cellprob = masks.squeeze(), dP.squeeze(), cellprob.squeeze()
        timings['squeeze'] = time.time() - t0

        # undo resizing:
        t0 = time.time()
        if image_scaling is not None or anisotropy is not None:

            dP = self._resize_gradients(dP, to_y_size=Ly_0, to_x_size=Lx_0, to_z_size=Lz_0) # works for 2 or 3D: 
            cellprob = self._resize_cellprob(cellprob, to_x_size=Lx_0, to_y_size=Ly_0, to_z_size=Lz_0)

            if do_3D:
                if compute_masks:
                    # Rescale xy then xz:
                    masks = transforms.resize_image(masks, Ly=Ly_0, Lx=Lx_0, no_channels=True, interpolation=cv2.INTER_NEAREST)
                    masks = masks.transpose(1, 0, 2)
                    masks = transforms.resize_image(masks, Ly=Lz_0, Lx=Lx_0, no_channels=True, interpolation=cv2.INTER_NEAREST)
                    masks = masks.transpose(1, 0, 2)

            else:
                # 2D or 3D stitching case:
                if compute_masks:
                    masks = transforms.resize_image(masks, Ly=Ly_0, Lx=Lx_0, no_channels=True, interpolation=cv2.INTER_NEAREST)
        timings['undo_resize'] = time.time() - t0

        timings['total'] = time.time() - t_start
        # models_logger.info(f"eval timings: {timings}")

        return masks, [plot.dx_to_circ(dP), dP, cellprob], styles
    

    def _resize_cellprob(self, prob: np.ndarray, to_y_size: int, to_x_size: int, to_z_size: int = None) -> np.ndarray:
        """
        Resize cellprob array to specified dimensions for either 2D or 3D.

        Parameters:
            prob (numpy.ndarray): The cellprobs to resize, either in 2D or 3D. Returns the same ndim as provided.
            to_y_size (int): The target size along the Y-axis.
            to_x_size (int): The target size along the X-axis.
            to_z_size (int, optional): The target size along the Z-axis. Required
                for 3D cellprobs.

        Returns:
            numpy.ndarray: The resized cellprobs array with the same number of dimensions
            as the input.

        Raises:
            ValueError: If the input cellprobs array does not have 3 or 4 dimensions.
        """
        prob_shape = prob.shape
        prob = prob.squeeze()
        squeeze_happened = prob.shape != prob_shape
        prob_shape = np.array(prob_shape)

        if prob.ndim == 2:
            # 2D case:
            prob = transforms.resize_image(prob, Ly=to_y_size, Lx=to_x_size, no_channels=True)
            if squeeze_happened:
                prob = np.expand_dims(prob, int(np.argwhere(prob_shape == 1))) # add back empty axis for compatibility
        elif prob.ndim == 3:
            # 3D case: 
            prob = transforms.resize_image(prob, Ly=to_y_size, Lx=to_x_size, no_channels=True)
            prob = prob.transpose(1, 0, 2)
            prob = transforms.resize_image(prob, Ly=to_z_size, Lx=to_x_size, no_channels=True)
            prob = prob.transpose(1, 0, 2)
        else:
            raise ValueError(f'gradients have incorrect dimension after squeezing. Should be 2 or 3, prob shape: {prob.shape}')
        
        return prob


    def _resize_gradients(self, grads: np.ndarray, to_y_size: int, to_x_size: int, to_z_size: int = None) -> np.ndarray:
        """
        Resize gradient arrays to specified dimensions for either 2D or 3D gradients.

        Parameters:
            grads (np.ndarray): The gradients to resize, either in 2D or 3D. Returns the same ndim as provided.
            to_y_size (int): The target size along the Y-axis.
            to_x_size (int): The target size along the X-axis.
            to_z_size (int, optional): The target size along the Z-axis. Required
                for 3D gradients.

        Returns:
            numpy.ndarray: The resized gradient array with the same number of dimensions
            as the input.

        Raises:
            ValueError: If the input gradient array does not have 3 or 4 dimensions.
        """
        grads_shape = grads.shape
        grads = grads.squeeze()
        squeeze_happened = grads.shape != grads_shape
        grads_shape = np.array(grads_shape)

        if grads.ndim == 3:
            # 2D case, with XY flows in 2 channels:
            grads = np.moveaxis(grads, 0, -1) # Put gradients last
            grads = transforms.resize_image(grads, Ly=to_y_size, Lx=to_x_size, no_channels=False)
            grads = np.moveaxis(grads, -1, 0) # Put gradients first

            if squeeze_happened:
                grads = np.expand_dims(grads, int(np.argwhere(grads_shape == 1))) # add back empty axis for compatibility
        elif grads.ndim == 4:
            # dP has gradients that can be treated as channels:
            grads = grads.transpose(1, 2, 3, 0) # move gradients last:
            grads = transforms.resize_image(grads, Ly=to_y_size, Lx=to_x_size, no_channels=False)
            grads = grads.transpose(1, 0, 2, 3) # switch axes to resize again
            grads = transforms.resize_image(grads, Ly=to_z_size, Lx=to_x_size, no_channels=False)
            grads = grads.transpose(3, 1, 0, 2) # undo transposition
        else:
            raise ValueError(f'gradients have incorrect dimension after squeezing. Should be 3 or 4, grads shape: {grads.shape}')
        
        return grads


    def _run_net(self, x, 
                augment=False, 
                batch_size=8, tile_overlap=0.1,
                bsize=256, anisotropy=1.0, do_3D=False):
        """ run network on image x """
        tic = time.time()
        shape = x.shape
        nimg = shape[0]


        if do_3D:
            Lz, Ly, Lx = shape[:-1]
            if anisotropy is not None and anisotropy != 1.0:
                models_logger.info(f"resizing 3D image with anisotropy={anisotropy}")
                x = transforms.resize_image(x.transpose(1,0,2,3),
                                        Ly=int(Lz*anisotropy), 
                                        Lx=int(Lx)).transpose(1,0,2,3)
            yf, styles = run_3D(self.net, x,
                                batch_size=batch_size, augment=augment,  
                                tile_overlap=tile_overlap, 
                                bsize=bsize
                                )
            cellprob = yf[..., -1]
            dP = yf[..., :-1].transpose((3, 0, 1, 2))
        else:
            yf, styles = run_net(self.net, x, bsize=bsize, augment=augment,
                                batch_size=batch_size,  
                                tile_overlap=tile_overlap, 
                                )
            cellprob = yf[..., -1]
            dP = yf[..., -3:-1].transpose((3, 0, 1, 2))
            if yf.shape[-1] > 3:
                styles = yf[..., :-3]
        
        styles = styles.squeeze()

        net_time = time.time() - tic
        if nimg > 1:
            models_logger.info("network run in %2.2fs" % (net_time))

        return dP, cellprob, styles
    
    def _compute_masks(self, shape, dP, cellprob, flow_threshold=0.4, cellprob_threshold=0.0,
                       min_size=15, max_size_fraction=0.4, niter=None,
                       do_3D=False, stitch_threshold=0.0):
        """ compute masks from flows and cell probability """
        changed_device_from = None
        if self.device.type == "mps" and do_3D:
            models_logger.warning("MPS does not support 3D post-processing, switching to CPU")
            self.device = torch.device("cpu")
            changed_device_from = "mps"
        Lz, Ly, Lx = shape[:3]
        tic = time.time()
        if do_3D:
            masks = dynamics.resize_and_compute_masks(
                dP, cellprob, niter=niter, cellprob_threshold=cellprob_threshold,
                flow_threshold=flow_threshold, do_3D=do_3D,
                min_size=min_size, max_size_fraction=max_size_fraction, 
                resize=shape[:3] if (np.array(dP.shape[-3:])!=np.array(shape[:3])).sum() 
                        else None,
                device=self.device)
        else:
            nimg = shape[0]
            Ly0, Lx0 = cellprob[0].shape 
            resize = None if Ly0==Ly and Lx0==Lx else [Ly, Lx]
            tqdm_out = utils.TqdmToLogger(models_logger, level=logging.INFO)
            iterator = trange(nimg, file=tqdm_out,
                            mininterval=30) if nimg > 1 else range(nimg)
            for i in iterator:
                # turn off min_size for 3D stitching
                min_size0 = min_size if stitch_threshold == 0 or nimg == 1 else -1
                outputs = dynamics.resize_and_compute_masks(
                    dP[:, i], cellprob[i],
                    niter=niter, cellprob_threshold=cellprob_threshold,
                    flow_threshold=flow_threshold, resize=resize,
                    min_size=min_size0, max_size_fraction=max_size_fraction,
                    device=self.device)
                if i==0 and nimg > 1:
                    masks = np.zeros((nimg, shape[1], shape[2]), outputs.dtype)
                if nimg > 1:
                    masks[i] = outputs
                else:
                    masks = outputs

            if stitch_threshold > 0 and nimg > 1:
                models_logger.info(
                    f"stitching {nimg} planes using stitch_threshold={stitch_threshold:0.3f} to make 3D masks"
                )
                masks = utils.stitch3D(masks, stitch_threshold=stitch_threshold)
                masks = utils.fill_holes_and_remove_small_masks(
                    masks, min_size=min_size)
            elif nimg > 1:
                models_logger.warning(
                    "3D stack used, but stitch_threshold=0 and do_3D=False, so masks are made per plane only"
                )

        flow_time = time.time() - tic
        if shape[0] > 1:
            models_logger.info("masks created in %2.2fs" % (flow_time))
        
        if changed_device_from is not None:
            models_logger.info("switching back to device %s" % self.device)
            self.device = torch.device(changed_device_from)
        return masks
