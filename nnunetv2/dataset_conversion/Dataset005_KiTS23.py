from batchgenerators.utilities.file_and_folder_operations import *
import shutil

from nnunetv2.paths import nnUNet_raw
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

import random

random.seed(42)




def convert_KiTS23(src_data_folder: str, dataset_id=5):

    

    # Arbitrary task id. This is just to ensure each dataset ha a unique number. Set this to whatever ([0-999]) you
    # want
    
    task_name = "KiTS23"

    foldername = "Dataset%03.0d_%s" % (dataset_id, task_name)

    # setting up nnU-Net folders
    out_base = join(nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)

    case_ids = subdirs(src_data_folder, prefix='case_', join=False)
    patients_train = sorted(case_ids)
    patients_test = random.sample(patients_train, 99)
    patients_train = [f for f in patients_train if f not in patients_test]
    num_training_cases = len(patients_train)

    for c in patients_train:
        shutil.copy(join(src_data_folder, c, 'segmentation.nii.gz'), join(labelstr, c + '.nii.gz'))
        shutil.copy(join(src_data_folder, c, 'imaging.nii.gz'), join(imagestr, c + '_0000.nii.gz'))
    
    for c in patients_test:
        shutil.copy(join(src_data_folder, c, 'segmentation.nii.gz'), join(labelstr, c + '.nii.gz'))
        shutil.copy(join(src_data_folder, c, 'imaging.nii.gz'), join(imagests, c + '_0000.nii.gz'))


    generate_dataset_json(
                          out_base, {0: "CT"},
                          labels = {
                              "background": 0,
                              "kidney": 1,
                              "tumor":2,
                              "cyst":3
                          },
                          file_ending=".nii.gz",
                          num_training_cases=num_training_cases,
                          license="CC-BY-NC-SA 4.0",
                          description='see https://kits23.kits-challenge.org/',
                          reference='https://kits23.kits-challenge.org/',
                          release='0')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_folder",
        type=str,
        help="The downloaded KiTS23 dataset dir. ",
    )
    parser.add_argument(
        "-d", "--dataset_id", required=False, type=int, default=5, help="nnU-Net Dataset ID, default: 5"
    )
    args = parser.parse_args()
    print("Converting...")
    convert_KiTS23(args.input_folder, args.dataset_id)
    print("Done!")
