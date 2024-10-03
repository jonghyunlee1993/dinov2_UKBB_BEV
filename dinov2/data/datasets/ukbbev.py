from enum import Enum
import os
from typing import Callable, Optional, Tuple, Union

import logging
import numpy as np
 
from .extended import ExtendedVisionDataset

logger = logging.getLogger("dinov2")


class _Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    @property
    def length(self) -> int:
        split_lengths = {
            # TODO: get these numbers from the dataset
            _Split.TRAIN: 703_510,
            _Split.VAL: 150_750,
            _Split.TEST: 150_750,
        }
        return split_lengths[self]

    def get_dirname(self, class_id: Optional[str] = None) -> str:
        return self.value if class_id is None else os.path.join(self.value, class_id)

    def get_image_relpath(self, eid: int, zoom_level: int) -> str:
        dirname = self.get_dirname()
        basename = f"bev_image_eid-{eid:07d}_zoom-{zoom_level}"
        return os.path.join(dirname, basename + ".png")

    def parse_image_relpath(self, image_relpath: str) -> Tuple[int, int]:
        assert self != _Split.TEST
        dirname, filename = os.path.split(image_relpath)
        basename, _ = os.path.splitext(filename)
        # Extract eid and zoom_level from the image filename
        parts = basename.split('_')
        eid = int(parts[2].split('-')[1])
        zoom_level = int(parts[3].split('-')[1])
        return eid, zoom_level


class UKBBEV(ExtendedVisionDataset):
    Target = Union[int]
    Split = Union[_Split]

    def __init__(
        self,
        *,
        split: "UKBBEV.Split",
        root: str,
        extra: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._extra_root = extra
        self._split = split

        self._entries = None

    @property
    def split(self) -> "UKBBEV.Split":
        return self._split

    @property
    def _entries_path(self) -> str:
        return f"entries-{self._split.value.upper()}.npy"

    def get_image_data(self, index: int) -> bytes:
        entries = self._get_entries()
        eid = entries[index]["eid"]
        zoom_level = entries[index]["zoom_level"]

        image_relpath = self.split.get_image_relpath(eid, zoom_level)
        image_full_path = os.path.join(self.root, image_relpath)
        with open(image_full_path, mode="rb") as f:
            image_data = f.read()
        return image_data

    def _get_entries(self) -> np.ndarray:
        if self._entries is None:
            self._entries = self._load_extra(self._entries_path)
        assert self._entries is not None
        return self._entries

    def __len__(self) -> int:
        entries = self._get_entries()
        return len(entries)

    def _get_extra_full_path(self, extra_path: str) -> str:
        return os.path.join(self._extra_root, extra_path)

    def _load_extra(self, extra_path: str) -> np.ndarray:
        extra_full_path = self._get_extra_full_path(extra_path)
        return np.load(extra_full_path, mmap_mode="r")

    def _save_extra(self, extra_array: np.ndarray, extra_path: str) -> None:
        extra_full_path = self._get_extra_full_path(extra_path)
        os.makedirs(self._extra_root, exist_ok=True)
        np.save(extra_full_path, extra_array)

    def get_target(self, index: int) -> int:
        # Return 0 for now because we don't have any labels
        return 0

    def dump_extra(self) -> None:
        self._dump_entries()

    def _dump_entries(self) -> None:
        # Implementation similar to ImageNet._dump_entries
        # Adjust for PNG files and UKBBEV specific structure
        split = self.split
        if split == UKBBEV.Split.TRAIN:
            dataset_root = os.path.join(self.root, split.get_dirname())
            dataset = os.listdir(dataset_root)
            sample_count = len(dataset)

            dtype = np.dtype(
                [
                    ("eid", "<u4"),
                    ("zoom_level", "<u4"),
                ]
            )
            entries_array = np.empty(sample_count, dtype=dtype)

            old_percent = -1
            for index in range(sample_count):
                percent = 100 * (index + 1) // sample_count
                if percent > old_percent:
                    logger.info(f"creating entries: {percent}%")
                    old_percent = percent

                image_full_path = dataset[index]
                image_relpath = os.path.relpath(image_full_path, self.root)
                eid, zoom_level = split.parse_image_relpath(image_relpath)
                entries_array[index] = (eid, zoom_level)

        logger.info(f'saving entries to "{self._entries_path}"')
        self._save_extra(entries_array, self._entries_path)
