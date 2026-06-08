import io
import json
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import PIL.Image


def load_h5_file(hf, path):
    # Helper function to load files from h5 file
    if path.endswith('.png'):
        rtn = np.array(PIL.Image.open(io.BytesIO(np.array(hf[path]))))
        rtn = rtn.reshape(*rtn.shape[:2], -1).transpose(2, 0, 1)
    elif path.endswith('.json'):
        rtn = json.loads(np.array(hf[path]).tobytes().decode('utf-8'))
    elif path.endswith('.npy'):
        rtn= np.array(hf[path])
    else:
        raise ValueError('Unknown file type: {}'.format(path))
    return rtn


class CustomINH5Dataset(Dataset):
    def __init__(self, data_dir):
        PIL.Image.init()
        supported_ext = PIL.Image.EXTENSION.keys() | {'.npy'}

        self.data_dir = data_dir
        self.h5_path = os.path.join(self.data_dir, "images.h5")
        self.h5_json_path = os.path.join(self.data_dir, "images_h5.json")
        self.h5f = h5py.File(self.h5_path, 'r')

        with open(self.h5_json_path, 'r') as f:
            self.h5_json = json.load(f)
        self.filelist = {fname for fname in self.h5_json}
        self.filelist = sorted(fname for fname in self.filelist if self._file_ext(fname) in supported_ext)

        labels = load_h5_file(self.h5f, "dataset.json")["labels"]
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self.filelist]
        labels = np.array(labels)
        self.labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])

    def __len__(self):
        return len(self.filelist)

    def _file_ext(self, fname):
        return os.path.splitext(fname)[1].lower()

    def __del__(self):
        self.h5f.close()

    def __getitem__(self, index):
        """
        Images should be '.png'
        """
        image_fname = self.filelist[index]
        image = load_h5_file(self.h5f, image_fname)
        return torch.from_numpy(image), torch.tensor(self.labels[index])


class CustomH5Dataset(Dataset):
    def __init__(self, data_dir, vae_latents_name="repae-invae-400k"):
        PIL.Image.init()
        supported_ext = PIL.Image.EXTENSION.keys() | {'.npy'}

        self.images_h5 = h5py.File(os.path.join(data_dir, 'images.h5'), "r")
        self.features_h5 = h5py.File(os.path.join(data_dir, f'{vae_latents_name}.h5'), "r")
        images_json = os.path.join(data_dir, 'images_h5.json')
        features_json = os.path.join(data_dir, f'{vae_latents_name}_h5.json')

        with open(images_json, 'r') as f:
            images_json = json.load(f)
        with open(features_json, 'r') as f:
            features_json = json.load(f)

        # images
        self._image_fnames = {fname for fname in images_json}
        self.image_fnames = sorted(fname for fname in self._image_fnames if self._file_ext(fname) in supported_ext)

        # features
        self._feature_fnames = {fname for fname in features_json}
        self.feature_fnames = sorted(fname for fname in self._feature_fnames if self._file_ext(fname) in supported_ext)
        
        # labels
        fname = 'dataset.json'
        labels = load_h5_file(self.features_h5, fname)['labels']
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self.feature_fnames]
        labels = np.array(labels)
        self.labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])

    def _file_ext(self, fname):
        return os.path.splitext(fname)[1].lower()

    def __len__(self):
        assert len(self.image_fnames) == len(self.feature_fnames), \
            "Number of feature files and label files should be same"
        return len(self.feature_fnames)

    def __del__(self):
        self.images_h5.close()
        self.features_h5.close()

    def __getitem__(self, idx):
        image_fname = self.image_fnames[idx]
        feature_fname = self.feature_fnames[idx]
        image_ext = self._file_ext(image_fname)

        image = load_h5_file(self.images_h5, image_fname)
        if image_ext == '.npy':
            # npy needs some extra care
            image = image.reshape(-1, *image.shape[-2:])

        features = load_h5_file(self.features_h5, feature_fname)
        return torch.from_numpy(image), torch.from_numpy(features), torch.tensor(self.labels[idx])


class CustomPairedH5Dataset(Dataset):
    """
    Dataset for paired image-to-image translation (e.g. HE ↔ IHC).

    Loads two H5 files (domain_a, domain_b) and returns matching image pairs
    by sorted filename order.  Both H5 files must contain the same number of
    samples with corresponding filenames at each index.

    Args:
        data_dir (str): Directory containing the two H5 files.
        h5_name_a (str): Basename (without .h5) for domain A (e.g. 'he').
        h5_name_b (str): Basename (without .h5) for domain B (e.g. 'ihc').

    Returns per item:
        image_a: torch.Tensor [C, H, W] uint8
        image_b: torch.Tensor [C, H, W] uint8
    """

    def __init__(
        self,
        data_dir: str,
        h5_name_a: str = "he",
        h5_name_b: str = "ihc",
    ):
        PIL.Image.init()
        supported_ext = PIL.Image.EXTENSION.keys() | {".npy"}

        self.data_dir = data_dir

        # Open H5 files for domain A and B
        self.h5a = h5py.File(os.path.join(data_dir, f"{h5_name_a}.h5"), "r")
        self.h5b = h5py.File(os.path.join(data_dir, f"{h5_name_b}.h5"), "r")

        json_a = os.path.join(data_dir, f"{h5_name_a}_h5.json")
        json_b = os.path.join(data_dir, f"{h5_name_b}_h5.json")

        with open(json_a, "r") as f:
            meta_a = json.load(f)
        with open(json_b, "r") as f:
            meta_b = json.load(f)

        self.fnames_a = sorted(
            fname for fname in meta_a if self._file_ext(fname) in supported_ext
        )
        self.fnames_b = sorted(
            fname for fname in meta_b if self._file_ext(fname) in supported_ext
        )

        assert len(self.fnames_a) == len(self.fnames_b), (
            f"Domain A ({len(self.fnames_a)}) and domain B ({len(self.fnames_b)}) "
            f"must have the same number of samples."
        )

    def _file_ext(self, fname):
        return os.path.splitext(fname)[1].lower()

    def __len__(self):
        return len(self.fnames_a)

    def __del__(self):
        if hasattr(self, "h5a"):
            self.h5a.close()
        if hasattr(self, "h5b"):
            self.h5b.close()

    def __getitem__(self, idx):
        image_a = load_h5_file(self.h5a, self.fnames_a[idx])
        image_b = load_h5_file(self.h5b, self.fnames_b[idx])
        return torch.from_numpy(image_a), torch.from_numpy(image_b)


class CSVPairedDataset(Dataset):
    """
    Dataset for paired image-to-image translation loading paths from a CSV file.
    Matches the BiBBDM CSV data-loading structure but returns format compatible
    with REPA-E (uint8 tensors).

    Args:
        csv_path (str): Path to the CSV file.
        image_size (int): Expected image resolution (e.g. 256).
    """

    def __init__(self, csv_path: str, image_size: int = 256, no_crop: bool = False, center_crop: bool = False):
        import pandas as pd
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.image_size = image_size
        self.no_crop = no_crop
        self.center_crop = center_crop

        # Check required columns
        required_cols = {"input_image_path", "target_image_path"}
        missing_cols = required_cols - set(self.df.columns)
        if missing_cols:
            raise ValueError(
                f"CSV file must contain columns: {required_cols}. Missing: {missing_cols}"
            )

        # Detect optional text columns for text conditioning
        self.has_text = "source_txt" in self.df.columns and "target_txt" in self.df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        source_path = row["input_image_path"]
        target_path = row["target_image_path"]

        # Load images
        try:
            image_a = PIL.Image.open(source_path).convert("RGB")
        except Exception as e:
            raise FileNotFoundError(f"Failed to load source image at '{source_path}': {e}")

        try:
            image_b = PIL.Image.open(target_path).convert("RGB")
        except Exception as e:
            raise FileNotFoundError(f"Failed to load target image at '{target_path}': {e}")

        # Crop / resize to expected size (skip when no_crop=True for inference)
        if not self.no_crop:
            w, h = image_a.size
            crop_size = self.image_size
            if w >= crop_size and h >= crop_size:
                if self.center_crop:
                    i = (h - crop_size) // 2
                    j = (w - crop_size) // 2
                else:
                    i = np.random.randint(0, h - crop_size + 1)
                    j = np.random.randint(0, w - crop_size + 1)
                image_a = image_a.crop((j, i, j + crop_size, i + crop_size))
                image_b = image_b.crop((j, i, j + crop_size, i + crop_size))
            else:
                image_a = image_a.resize((crop_size, crop_size), PIL.Image.BICUBIC)
                image_b = image_b.resize((crop_size, crop_size), PIL.Image.BICUBIC)

        # Convert to numpy array [H, W, C] and transpose to [C, H, W]
        arr_a = np.array(image_a).transpose(2, 0, 1)
        arr_b = np.array(image_b).transpose(2, 0, 1)

        if self.has_text:
            source_txt = str(row["source_txt"]) if not (isinstance(row["source_txt"], float) and np.isnan(row["source_txt"])) else ""
            target_txt = str(row["target_txt"]) if not (isinstance(row["target_txt"], float) and np.isnan(row["target_txt"])) else ""
            return torch.from_numpy(arr_a), torch.from_numpy(arr_b), source_txt, target_txt
        else:
            return torch.from_numpy(arr_a), torch.from_numpy(arr_b)


