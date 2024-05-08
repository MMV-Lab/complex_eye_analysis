import numpy as np
from aicsimageio import AICSImage
from pathlib import Path
from tqdm import tqdm

save_zarr = True  # Flag to save zarr files, False/True
save_raw_tiff = False  # Flag to save raw tiff files, False/True


if save_zarr:
    import zarr
    Path("./data/zarrs/").mkdir(parents=True, exist_ok=True)    

if save_raw_tiff:
    from aicsimageio.writers import OmeTiffWriter
    Path("./data/raw_tiff/").mkdir(parents=True, exist_ok=True)

data_path = "./data/raw/"
files = Path(data_path).glob("*")


for file in tqdm(files):
    reader_raw = AICSImage(file)
    if reader_raw.dims.T > 1:
        img_raw = reader_raw.get_image_data("TYX")
    else:
        img_raw = reader_raw.get_image_data("ZYX")
    if save_raw_tiff:
        OmeTiffWriter.save(
            img_raw,
            "./data/raw_tiff/" + file.with_suffix(".tiff").name,
            dim_order="TYX",
        )

    if save_zarr:
        reader_seg = AICSImage("./data/segmentation/" + file.with_suffix(".tiff").name)
        if reader_seg.dims.T > 1:
            img_seg = reader_seg.get_image_data("TYX")
        else:
            img_seg = reader_seg.get_image_data("ZYX")

        tracks = np.load("./data/tracks/" + file.stem + "_trackslayer.npy")

        root = zarr.open("./data/zarrs/" + file.with_suffix(".zarr").name, mode="w")
        r1 = root.create_dataset("raw_data", shape=img_raw.shape, dtype="f8", data=img_raw)
        s1 = root.create_dataset(
            "segmentation_data", shape=img_seg.shape, dtype="i4", data=img_seg
        )
        t1 = root.create_dataset(
            "tracking_data", shape=tracks.shape, dtype="i4", data=tracks
        )
