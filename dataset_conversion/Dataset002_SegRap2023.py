from collections import OrderedDict
from nnunetv2.paths import nnUNet_raw
from batchgenerators.utilities.file_and_folder_operations import *
import shutil
import SimpleITK as sitk
from pathlib import Path
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
import os



import random

random.seed(42)



def make_out_dirs(dataset_id: int, task_name="SegRap2023"):
    dataset_name = f"Dataset{dataset_id:03d}_{task_name}"

    out_dir = Path(nnUNet_raw.replace('"', "")) / dataset_name
    out_train_dir = out_dir / "imagesTr"
    out_labels_dir = out_dir / "labelsTr"
    out_test_dir = out_dir / "imagesTs"

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_train_dir, exist_ok=True)
    os.makedirs(out_labels_dir, exist_ok=True)
    os.makedirs(out_test_dir, exist_ok=True)

    return out_dir, out_train_dir, out_labels_dir, out_test_dir


def copy_files(src_data_folder: Path, train_dir: Path, labels_dir: Path, test_dir: Path):
    """Copy files from the SegRap dataset to the nnUNet dataset folder. Returns the number of training cases."""
    patients_train = sorted([f for f in (src_data_folder / "SegRap2023_Training_Set_120cases").iterdir() if f.is_dir()])
    patients_test = random.sample(patients_train, 25)
    patients_train = [f for f in patients_train if f not in patients_test]

    labels = src_data_folder / "SegRap2023_Training_Set_120cases_OneHot_Labels" / "Task002"

    # patients_test = sorted([f for f in (src_data_folder / "testing").iterdir() if f.is_dir()])

    num_training_cases = 0
    # Copy training files 
    for patient_dir in patients_train:
        for file in patient_dir.iterdir():
            if file.suffix == ".gz" and file.name == "image.nii.gz":
                shutil.copy(file, train_dir / f"{patient_dir.name}_0000.nii.gz")
                num_training_cases += 1
            elif file.suffix == ".gz" and file.name == "image_contrast.nii.gz":
                shutil.copy(file, train_dir / f"{patient_dir.name}_0001.nii.gz")
    
    # Copy test labels
    for patient_dir in patients_test:
        for file in patient_dir.iterdir():
            if file.suffix == ".gz" and file.name == "image.nii.gz":
                shutil.copy(file, test_dir / f"{patient_dir.name}_0000.nii.gz")
            elif file.suffix == ".gz" and file.name == "image_contrast.nii.gz":
                shutil.copy(file, test_dir / f"{patient_dir.name}_0001.nii.gz")
    
    # copy segmentations
    label_files = [i for i in subfiles(labels, suffix='.nii.gz', join=False) if
                         not i.startswith('.') and not i.startswith('_')]
    for f in label_files:
        shutil.copy(join(labels, f), join(labels_dir, f))
    
    return num_training_cases

    
def convert_SegRap2023(src_data_folder: str, dataset_id=2):
    out_dir, train_dir, labels_dir, test_dir = make_out_dirs(dataset_id=dataset_id)
    num_training_cases = copy_files(Path(src_data_folder), train_dir, labels_dir, test_dir)

    generate_dataset_json(
        str(out_dir),
        channel_names={
            0: "non_contrast_CT",
            1: "contrast_CT"
        },
        labels={
            "background": 0,
            "GTVp": 1,
            "GTVnd": 2
        },
        file_ending=".nii.gz",
        num_training_cases=num_training_cases,
        reference = "https://segrap2023.grand-challenge.org/",
        release = "1.0 25/04/2023",
        license = "CC-BY-SA 4.0",
        description= "Segmentation of Organs-at-Risk and Gross Tumor Volume of NPC for Radiotherapy Planning (SegRap2023)",
    )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_folder",
        type=str,
        help="The downloaded SegRap2023 dataset dir. ",
    )
    parser.add_argument(
        "-d", "--dataset_id", required=False, type=int, default=2, help="nnU-Net Dataset ID, default: 2"
    )
    args = parser.parse_args()
    print("Converting...")
    convert_SegRap2023(args.input_folder, args.dataset_id)
    print("Done!")

    
    

                
            
