import os
from pathlib import Path
from typing import Union
import json
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from skimage.transform import resize
from typing import Callable, Optional, Union, List
'''
@inproceedings{ling_temporal_2023,
   title = {Extraction of {Volumetric} {Indices} from {Echocardiography}: {Which} {Deep} {Learning} {Solution} for {Clinical} {Use}?},
   doi = {10.1007/978-3-031-35302-4_25},
   series = {Lecture {Notes} in {Computer} {Science}},
   booktitle = {Functional {Imaging} and {Modeling} of the {Heart}},
   publisher = {Springer Nature Switzerland},
   author = {Ling, Hang Jung and Painchaud, Nathan and Courand, Pierre-Yves and Jodoin, Pierre-Marc and Garcia, Damien and Bernard, Olivier},
   month = june,
   year = {2023},
   pages = {245--254},
}
'''
def save_json(obj, file: str, indent: int = 4, sort_keys: bool = True) -> None:
    with open(file, "w") as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)

def subfiles(
    folder: str, join: bool = True, prefix: str = None, suffix: str = None, sort: bool = True
) -> List[str]:
    if join:
        l = os.path.join  # noqa: E741
    else:
        l = lambda x, y: y  # noqa: E731, E741
    res = [
        l(folder, i)
        for i in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, i))
        and (prefix is None or i.startswith(prefix))
        and (suffix is None or i.endswith(suffix))
    ]
    if sort:
        res.sort()
    return res

def get_identifiers_from_split_files(folder: str) -> np.ndarray:
    """Get unique case identifiers from split imagesTr or imagesTs folders.

    Args:
        folder: Path to folder containing train or test images.

    Returns:
        Sorted unique case identifiers array.
    """
    uniques = np.unique([i[:-12] for i in subfiles(folder, suffix=".nii.gz", join=False)])
    return uniques


def generate_dataset_json(
    output_file: str,
    imagesTr_dir: str,
    imagesTs_dir: Optional[str],
    modalities: tuple[str, ...],
    labels: dict[int, str],
    dataset_name: str,
    sort_keys: bool = True,
    license: Optional[str] = "hands off!",
    dataset_description: Optional[str] = "",
    dataset_reference: Optional[str] = "",
    dataset_release: Optional[str] = "0.0",
) -> None:
    """Generate dataset.json file.

    Args:
        output_file: Full path to the dataset.json you intend to write, so output_file='DATASET_PATH/dataset.json'
            where the folder DATASET_PATH points to is the one with the imagesTr and labelsTr subfolders.
        imagesTr_dir: Path to the imagesTr folder of that dataset.
        imagesTs_dir: Path to the imagesTs folder of that dataset. Can be None
        modalities: Tuple of strings with modality names. Must be in the same order as the images
            (first entry corresponds to _0000.nii.gz, etc). Example: ('T1', 'T2', 'FLAIR').
        labels: Dict mapping the label IDs to label names. Note that 0 is always supposed to be background!
            Example: {0: 'background', 1: 'edema', 2: 'enhancing tumor'}.
        dataset_name: Name of the dataset.
        sort_keys: Whether to sort the keys in dataset.json.
        license: License of the dataset.
        dataset_description: Quick description of the dataset.
        dataset_reference: Website of the dataset, if available.
        dataset_release: Version of the dataset.
    """
    train_identifiers = get_identifiers_from_split_files(imagesTr_dir)

    if imagesTs_dir is not None:
        test_identifiers = get_identifiers_from_split_files(imagesTs_dir)
    else:
        test_identifiers = []

    json_dict = {}
    json_dict["name"] = dataset_name
    json_dict["description"] = dataset_description
    json_dict["tensorImageSize"] = "4D"
    json_dict["reference"] = dataset_reference
    json_dict["licence"] = license
    json_dict["release"] = dataset_release
    json_dict["modality"] = {str(i): modalities[i] for i in range(len(modalities))}
    json_dict["labels"] = {str(i): labels[i] for i in labels.keys()}

    json_dict["numTraining"] = len(train_identifiers)
    json_dict["numTest"] = len(test_identifiers)
    json_dict["training"] = [
        {"image": "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i}
        for i in train_identifiers
    ]
    json_dict["test"] = ["./imagesTs/%s.nii.gz" % i for i in test_identifiers]

    if not output_file.endswith("dataset.json"):
        print(
            "WARNING: output file name is not dataset.json! This may be intentional or not. You decide. "
            "Proceeding anyways..."
        )
    save_json(json_dict, os.path.join(output_file), sort_keys=sort_keys)

def resample_image(
    image: np.ndarray,
    new_shape: Union[list, tuple],
    anisotropy_flag: bool,
    lowres_axis: Optional[np.ndarray] = None,
    interp_order: int = 3,
    order_z: int = 0,
) -> np.ndarray:
    """Resample an image.

    Args:
        image: Image numpy array to be resampled.
        new_shape: Shape after resampling.
        anisotropy_flag: Whether the image is anisotropic.
        lowres_axis: Axis of lowest resolution.
        interp_order: Interpolation order of skimage.transform.resize.
        order_z: Interpolation order for the lowest resolution axis in case of anisotropic image.

    Returns:
        Resampled image.
    """
    dtype_data = image.dtype
    shape = np.array(image[0].shape)
    if not np.all(shape == np.array(new_shape)):
        image = image.astype(float)
        resized_channels = []
        if anisotropy_flag:
            print("Anisotropic image, using separate z resampling")
            axis = lowres_axis[0]
            if axis == 0:
                new_shape_2d = new_shape[1:]
            elif axis == 1:
                new_shape_2d = new_shape[[0, 2]]
            else:
                new_shape_2d = new_shape[:-1]
            for image_c in image:
                resized_slices = []
                for i in range(shape[axis]):
                    if axis == 0:
                        image_c_2d_slice = image_c[i]
                    elif axis == 1:
                        image_c_2d_slice = image_c[:, i]
                    else:
                        image_c_2d_slice = image_c[:, :, i]
                    image_c_2d_slice = resize(
                        image_c_2d_slice,
                        new_shape_2d,
                        order=interp_order,
                        mode="edge",
                        cval=0,
                        clip=True,
                        anti_aliasing=False,
                    )
                    resized_slices.append(image_c_2d_slice.astype(dtype_data))
                resized = np.stack(resized_slices, axis=axis)
                if not shape[axis] == new_shape[axis]:
                    resized = resize(
                        resized,
                        new_shape,
                        order=order_z,
                        mode="constant",
                        cval=0,
                        clip=True,
                        anti_aliasing=False,
                    )
                resized_channels.append(resized.astype(dtype_data))
        else:
            print("Not using separate z resampling")
            for image_c in image:
                resized = resize(
                    image_c,
                    new_shape,
                    order=interp_order,
                    mode="edge",
                    cval=0,
                    clip=True,
                    anti_aliasing=False,
                )
                resized_channels.append(resized.astype(dtype_data))
        reshaped = np.stack(resized_channels, axis=0)
        return reshaped.astype(dtype_data)
    else:
        print("No resampling necessary")
        return image


def resample_label(
    label: np.ndarray,
    new_shape: Union[list, tuple],
    anisotropy_flag: bool,
    lowres_axis: Optional[np.ndarray] = None,
    interp_order: int = 1,
    order_z: int = 0,
) -> np.ndarray:
    """Resample a label.

    Args:
        label: Label numpy array to be resampled.
        new_shape: Shape after resampling.
        anisotropy_flag: Whether the label is anisotropic.
        lowres_axis: Axis of lowest resolution.
        interp_order: Interpolation order of skimage.transform.resize.
        order_z: Interpolation order for the lowest resolution axis in case of anisotropic label.

    Returns:
        Resampled label.
    """
    shape = np.array(label[0].shape)
    if not np.all(shape == np.array(new_shape)):
        reshaped = np.zeros(new_shape, dtype=np.uint8)
        n_class = np.max(label)
        if anisotropy_flag:
            print("Anisotropic label, using separate z resampling")
            axis = lowres_axis[0]
            depth = shape[axis]
            if axis == 0:
                new_shape_2d = new_shape[1:]
                reshaped_2d = np.zeros((depth, *new_shape_2d), dtype=np.uint8)
            elif axis == 1:
                new_shape_2d = new_shape[[0, 2]]
                reshaped_2d = np.zeros((new_shape_2d[0], depth, new_shape_2d[1]), dtype=np.uint8)
            else:
                new_shape_2d = new_shape[:-1]
                reshaped_2d = np.zeros((*new_shape_2d, depth), dtype=np.uint8)

            for class_ in range(1, int(n_class) + 1):
                for depth_ in range(depth):
                    if axis == 0:
                        mask = label[0, depth_] == class_
                    elif axis == 1:
                        mask = label[0, :, depth_] == class_
                    else:
                        mask = label[0, :, :, depth_] == class_
                    resized_2d = resize(
                        mask.astype(float),
                        new_shape_2d,
                        order=interp_order,
                        mode="edge",
                        cval=0,
                        clip=True,
                        anti_aliasing=False,
                    )
                    if axis == 0:
                        reshaped_2d[depth_][resized_2d >= 0.5] = class_
                    elif axis == 1:
                        reshaped_2d[:, depth_][resized_2d >= 0.5] = class_
                    else:
                        reshaped_2d[:, :, depth_][resized_2d >= 0.5] = class_

            if not shape[axis] == new_shape[axis]:
                for class_ in range(1, int(n_class) + 1):
                    mask = reshaped_2d == class_
                    resized = resize(
                        mask.astype(float),
                        new_shape,
                        order=order_z,
                        mode="constant",
                        cval=0,
                        clip=True,
                        anti_aliasing=False,
                    )
                    reshaped[resized >= 0.5] = class_
            else:
                reshaped = reshaped_2d.astype(np.uint8)
        else:
            print("Not using separate z resampling")
            for class_ in range(1, int(n_class) + 1):
                mask = label[0] == class_
                resized = resize(
                    mask.astype(float),
                    new_shape,
                    order=interp_order,
                    mode="edge",
                    cval=0,
                    clip=True,
                    anti_aliasing=False,
                )
                reshaped[resized >= 0.5] = class_

        reshaped = np.expand_dims(reshaped, 0)
        return reshaped
    else:
        print("No resampling necessary")
        return label


def convert_to_nnUNet(
    data_dir: Union[Path, str],
    output_dir: Union[Path, str],
    test: bool = False,
    sequence: bool = False,
    views: list = ["2CH", "4CH"],
    resize: bool = False,
) -> None:
    """Convert Camus dataset to nnUNet's format.

    Args:
        data_dir: Path to the dataset.
        output_dir: Path to the output folder to save the converted data.
        test: Whether is test dataset.
        sequence: Whether to convert the whole sequence or 2CH/4CH ED/ES only. (images only)
        views: Views to be converted.
        resize: Whether to resize images to 256x256xT.
    """
    if not test:
        images_out_dir = os.path.join(output_dir, "imagesTr")
        labels_out_dir = os.path.join(output_dir, "labelsTr")
    else:
        if not resize:
            images_out_dir = os.path.join(output_dir, "imagesTs")
            labels_out_dir = os.path.join(output_dir, "labelsTs")
        else:
            images_out_dir = os.path.join(output_dir, "imagesTs256")
            labels_out_dir = os.path.join(output_dir, "labelsTs256")

    os.makedirs(images_out_dir, exist_ok=True)
    os.makedirs(labels_out_dir, exist_ok=True)

    for case in tqdm(os.listdir(data_dir)):
        case_path = os.path.join(data_dir, case)
        if os.listdir(case_path):
            if not sequence:
                for view in views:
                    for instant in ["ED", "ES"]:
                        case_identifier = f"{case}_{view}_{instant}"
                        image = sitk.ReadImage(os.path.join(case_path, f"{case_identifier}.nii.gz"))
                        if os.path.isfile(os.path.join(case_path, f"{case_identifier}_gt.nii.gz")):
                            label = sitk.ReadImage(
                                os.path.join(case_path, f"{case_identifier}_gt.nii.gz")
                            )
                        else:
                            label = None
                        sitk.WriteImage(
                            image, os.path.join(images_out_dir, f"{case_identifier}_0000.nii.gz")
                        )
                        if label is not None:
                            sitk.WriteImage(
                                label, os.path.join(labels_out_dir, f"{case_identifier}.nii.gz")
                            )
            else:
                for view in views:
                    case_identifier = f"{case}_{view}_sequence"
                    image = sitk.ReadImage(os.path.join(case_path, f"{case_identifier}.nii.gz"))
                    sitk.WriteImage(
                        image, os.path.join(images_out_dir, f"{case_identifier}_0000.nii.gz")
                    )
                    if os.path.isfile(os.path.join(case_path, f"{case_identifier}_gt.nii.gz")):
                        label = sitk.ReadImage(
                            os.path.join(case_path, f"{case_identifier}_gt.nii.gz")
                        )
                    else:
                        label = None
                    if resize:
                        ori_shape = image.GetSize()
                        ori_spacing = image.GetSpacing()
                        new_shape = [256, 256, ori_shape[-1]]
                        new_spacing = (
                            np.array(ori_spacing) * np.array(ori_shape) / np.array(new_shape)
                        )
                        image_array = sitk.GetArrayFromImage(image).transpose(2, 1, 0)
                        image_array = image_array[None]
                        resized_image_array = resample_image(image_array, new_shape, True)
                        image = sitk.GetImageFromArray(resized_image_array[0].transpose(2, 1, 0))
                        image.SetSpacing(new_spacing)

                    sitk.WriteImage(
                        image, os.path.join(images_out_dir, f"{case_identifier}_0000.nii.gz")
                    )
                    if label is not None:
                        if resize:
                            label_array = sitk.GetArrayFromImage(label).transpose(2, 1, 0)
                            label_array = label_array[None]
                            resized_label_array = resample_label(label_array, new_shape, True)
                            label = sitk.GetImageFromArray(
                                resized_label_array[0].transpose(2, 1, 0)
                            )
                            label.SetSpacing(new_spacing)
                        sitk.WriteImage(
                            label, os.path.join(labels_out_dir, f"{case_identifier}.nii.gz")
                        )


# def convert_to_CAMUS_submission(
#     predictions_dir: Union[Path, str], output_dir: Union[Path, str]
# ) -> None:
#     """Convert predictions to correct format for submission.

#     Args:
#         predictions_dir: Path to the prediction folder.
#         output_dir: Path to the output folder to save the converted predictions.
#     """
#     os.makedirs(output_dir, exist_ok=True)
#     for case in tqdm(os.listdir(predictions_dir)):
#         case_path = os.path.join(predictions_dir, case)
#         case_identifier = case[:-7]
#         image = sitk.ReadImage(case_path)
#         sitk.WriteImage(image, os.path.join(output_dir, f"{case_identifier}.mhd"))


if __name__ == "__main__":
    base = "/staff/wangtiantong/self-test/dataset/CAMUS_public/splited/train"
    test_data = "/staff/wangtiantong/self-test/dataset/CAMUS_public/splited/test"
    output_dir = "/staff/wangtiantong/nnU-Net/nnUNet/nnUNetFrame/dataset/nnUNet_raw/Dataset101_CAMUS"
    os.makedirs(output_dir, exist_ok=True)

    dataset_name = "CAMUS"

    imagesTr = os.path.join(output_dir, "imagesTr")
    labelsTr = os.path.join(output_dir, "labelsTr")
    imagesTs = os.path.join(output_dir, "imagesTs")
    labelsTs = os.path.join(output_dir, "labelsTs")

    os.makedirs(imagesTr, exist_ok=True)
    os.makedirs(labelsTr, exist_ok=True)
    os.makedirs(imagesTs, exist_ok=True)
    os.makedirs(labelsTs, exist_ok=True)

    # Convert train data to nnUNet's format
    convert_to_nnUNet(base, output_dir)

    # Generate dataset.json
    generate_dataset_json(
        os.path.join(output_dir, "dataset.json"),
        imagesTr,
        imagesTs,
        ("US",),
        {0: "background", 1: "LV", 2: "MYO", 3: "LA"},
        dataset_name,
    )

    # Convert test data to nnUNet's format
    convert_to_nnUNet(test_data, output_dir, sequence=False, test=True, resize=False)

    # # Convert predictions in Nifti format to raw/mhd
    # prediction_dir = "C:/Users/ling/Desktop/camus_test/inference_raw"
    # submission_dir = "C:/Users/ling/Desktop/camus_test/submission"
    # convert_to_CAMUS_submission(prediction_dir, submission_dir)