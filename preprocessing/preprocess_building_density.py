import os
from glob import glob

import buteo as beo
import numpy as np
from tqdm import tqdm

TILE_PATHS = [
    { "path": "/data_raw/egypt/", "name": "EGY1" },
    { "path": "/data_raw/ghana/", "name": "GHA1" },
    { "path": "/data_raw/israel_gaza_1/", "name": "ISR1" },
    { "path": "/data_raw/israel_gaza_2/", "name": "ISR2" },
    { "path": "/data_raw/tanzania_dar/", "name": "TZA1" },
    { "path": "/data_raw/tanzania_kigoma/", "name": "TZA2" },
    { "path": "/data_raw/tanzania_kilimanjaro/", "name": "TZA3" },
    { "path": "/data_raw/tanzania_mwanza_Q2/", "name": "TZA4" },
    { "path": "/data_raw/tanzania_mwanza_Q3/", "name": "TZA5" },
    { "path": "/data_raw/uganda/", "name": "UGA1" },
    { "path": "/data_raw/denmark_2020/", "name": "DNK1" },
    { "path": "/data_raw/denmark_2021/", "name": "DNK2" },
]

TEST_LOCATIONS = [
    "DNK1_53", "DNK2_53", "DNK1_45", "DNK2_45", "DNK1_42", "DNK2_42",
    "DNK1_17", "DNK2_17", "DNK1_21", "DNK2_21", "DNK1_58", "DNK2_58",
    "DNK1_48", "DNK2_48", "DNK1_60", "DNK2_60", "DNK1_51", "DNK2_51",
    "ISR1_1", "ISR2_1",
    "EGY1_51", "EGY1_73", "EGY1_69", "EGY1_71", "EGY1_42", "EGY1_37",
    "EGY1_14", "EGY1_57",
    "TZA1_2", "TZA3_4", "UGA1_7",
    "GHA1_159", "GHA1_113", "GHA1_189", "GHA1_187", "GHA1_229",
    "GHA1_221", "GHA1_118", "GHA1_88", "GHA1_216", "GHA1_240",
    "GHA1_140", "GHA1_12", "GHA1_15", "GHA1_167", "GHA1_102",
    "GHA1_38", "GHA1_190", "GHA1_28", "GHA1_77", "GHA1_209",
]


def normalize_y(y: np.ndarray) -> np.ndarray:
    y = y.astype(np.float32, copy=False)
    y_norm = np.empty_like(y, dtype=np.float32)
    np.divide(y, 100.0, out=y_norm)
    return y_norm


def preprocess_image_to_patches(
        folder_src: str,
        folder_dst: str,
        overlaps: int = 1,
        patch_size: int = 64,
        val_split_ratio: float = 0.1,
        building_density_ratio: float = 0.05,
        test_locations: list = None,
        train_locations: list = None,
        normalize_label: bool = True
) -> None:

    if train_locations is None:
        # If empty, all locations are used.
        train_locations = []

    if test_locations is None:
        test_locations = TEST_LOCATIONS

    for x in train_locations:
        if x in test_locations:
            raise ValueError("Location in both train and test.")

    images = os.listdir(folder_src)

    total = len(test_locations) + len(train_locations)
    if len(train_locations) == 0:
        total = int(len(images) / 4)

    processed = 0
    for img in images:
        if "mask" not in img:
            continue

        location = img.split("_")[0]
        fid = img.split("_")[1]

        path_mask = os.path.join(folder_src, f"{location}_{fid}_mask.tif")
        path_label = os.path.join(folder_src, f"{location}_{fid}_label.tif")
        path_s2 = os.path.join(folder_src, f"{location}_{fid}_s2.tif")

        if fid in test_locations:
            pass
        elif fid in train_locations:
            pass
        elif len(train_locations) == 0:
            pass
        else:
            processed += 1
            continue

        if f"{location}_{fid}" in test_locations:
            TARGET = "test"
        else:
            TARGET = "train"

        mask_arr = beo.raster_to_array(path_mask)
        metadata = beo.raster_to_metadata(path_mask)

        if metadata["height"] < patch_size or metadata["width"] < patch_size:
            processed += 1
            continue

        initial_patches = beo.array_to_patches(
            mask_arr,
            tile_size=patch_size,
            n_offsets=overlaps,
            border_check=True,
        )

        # Mask any tiles with any masked values.
        mask_bool = ~np.any(initial_patches == 0, axis=(1, 2, 3))
        mask_bool_sum = mask_bool.sum()
        mask_random = np.random.permutation(np.arange(mask_bool_sum))

        patches_label = beo.array_to_patches(
            beo.raster_to_array(path_label),
            tile_size=patch_size,
            n_offsets=overlaps,
            border_check=True,
        )[mask_bool][mask_random]

        # check which patches contain regions of interest
        patches_shape = patches_label.shape
        # reshape mask array to shape (num_patches, tile_size*tile_size)
        labels = patches_label.reshape(patches_shape[0], patches_shape[1] * patches_shape[2])
        # get percentage of buildings in each patch
        p_labels = (labels.sum(axis=-1) / 100) / (patches_shape[1] * patches_shape[2])
        # only patches with 20% pixels in the region of interest are considered as valid
        p_idx = np.asarray(p_labels >= building_density_ratio).nonzero()

        patches_label = patches_label[p_idx]
        idx_val = int(patches_label.shape[0] * (1 - val_split_ratio))

        if normalize_label:
            patches_label = normalize_y(patches_label)

        if TARGET == "train":
            patches_y_val = patches_label[idx_val:]
            patches_label = patches_label[:idx_val]

            if patches_y_val.shape[0] != 0:
                np.save(os.path.join(folder_dst, f"{location}_{fid}_val_y.npy"), patches_y_val)

        np.save(os.path.join(folder_dst, f"{location}_{fid}_{TARGET}_y.npy"), patches_label)

        patches_s2 = beo.array_to_patches(
            beo.raster_to_array(path_s2),
            tile_size=patch_size,
            n_offsets=overlaps,
            border_check=True,
        )[mask_bool][mask_random]

        patches_s2 = patches_s2[p_idx]

        if TARGET == "train":
            patches_s2_val = patches_s2[idx_val:]
            patches_s2 = patches_s2[:idx_val]

            if patches_s2_val.shape[0] != 0:
                np.save(os.path.join(folder_dst, f"{location}_{fid}_val_s2.npy"), patches_s2_val)

        np.save(os.path.join(folder_dst, f"{location}_{fid}_{TARGET}_s2.npy"), patches_s2)

        assert patches_y_val.shape[0] == patches_s2_val.shape[0], "Number of patches do not match."
        assert patches_label.shape[0] == patches_s2.shape[0], "Number of patches do not match."

        processed += 1
        print(f"Processed {location}_{fid} ({processed}/{total}).")


def preprocess_tile_to_image(folder_src: str,
                             folder_dst: str,
                             path_dict: dict=None) -> None:

    if path_dict is None:
        path_dict = TILE_PATHS

    for tile_dict in path_dict:
        folder = os.path.join(folder_src, (tile_dict["path"]))
        name = tile_dict["name"]

        path_mask = os.path.join(folder, "mask.gpkg")

        split_geoms = beo.vector_split_by_fid(path_mask)

        for geom in tqdm(sorted(split_geoms), total=len(split_geoms)):
            fid = os.path.splitext(os.path.basename(geom))[0].split("_")[-1]

            out_mask = beo.raster_clip(
                os.path.join(folder, "mask.tif"),
                clip_geom=geom,
                dst_nodata=None,
                out_path=os.path.join(folder_dst, f"{name}_{fid}_mask.tif"),
            )

            out_label = beo.raster_clip(
                os.path.join(folder, "labels.tif"),
                clip_geom=geom,
                dst_nodata=None,
                out_path=os.path.join(folder_dst, f"{name}_{fid}_label.tif"),
            )

            s2_clipped = beo.raster_clip(
                [
                    os.path.join(folder, "B02.tif"),
                    os.path.join(folder, "B03.tif"),
                    os.path.join(folder, "B04.tif"),
                    os.path.join(folder, "B08.tif"),
                    os.path.join(folder, "B05.tif"),
                    os.path.join(folder, "B06.tif"),
                    os.path.join(folder, "B07.tif"),
                    os.path.join(folder, "B8A.tif"),
                    os.path.join(folder, "B11.tif"),
                    os.path.join(folder, "B12.tif"),
                ],
                clip_geom=geom,
                dst_nodata=None,
            )
            out_s2 = beo.raster_stack_list(
                s2_clipped,
                out_path=os.path.join(folder_dst, f"{name}_{fid}_s2.tif"),
            )
            beo.delete_dataset_if_in_memory(s2_clipped)

            assert beo.check_rasters_are_aligned(
                [out_s2, out_mask, out_label],
                same_nodata=True,
            ), "Rasters are not aligned."

        beo.delete_dataset_if_in_memory_list(split_geoms)



if __name__ == '__main__':
    print('Processing tiles into images please wait...\n')
    preprocess_tile_to_image(folder_src="/home/lcamilleri/data/s12_buildings/",
                             folder_dst="/home/lcamilleri/data/s12_buildings/data_images/")

    print('\nProcessing images into patches please wait...\n')
    preprocess_image_to_patches(folder_src="/home/lcamilleri/data/s12_buildings/data_images/",
                                folder_dst="/home/lcamilleri/data/s12_buildings/data_patches/")

    images = glob("/home/lcamilleri/data/s12_buildings/data_images/" + "*_mask.tif")

    beo.vector_merge_features(
        beo.raster_get_footprints(images, out_format="geojson"),
        out_path=os.path.join("/home/lcamilleri/data/s12_buildings/", "data_extents.geojson"),
    )