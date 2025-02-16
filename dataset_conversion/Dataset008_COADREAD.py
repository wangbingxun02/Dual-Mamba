from batchgenerators.utilities.file_and_folder_operations import *
import shutil

from nnunetv2.paths import nnUNet_raw
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

import random
import glob

random.seed(10)




def convert_COADREAD(src_data_folder: str, dataset_id=8):

    

    # Arbitrary task id. This is just to ensure each dataset ha a unique number. Set this to whatever ([0-999]) you
    # want
    
    task_name = "COADREAD3"

    foldername = "Dataset%03.0d_%s" % (dataset_id, task_name)

    # setting up nnU-Net folders
    out_base = join(nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)

    case_ids = subdirs(src_data_folder,join=False)
    print(case_ids)
    patients_train = sorted(case_ids)
    patients_test = random.sample(patients_train, 15)
    patients_train = [f for f in patients_train if f not in patients_test]
    num_training_cases = len(patients_train)

    for c in patients_train:
        tr_files_1 = glob.glob(join(src_data_folder,c, "S*"))
        print(tr_files_1)
        shutil.copy(join(src_data_folder, c, c + '.nii.gz'), join(labelstr, c + '.nii.gz'))
        for file_path in tr_files_1:
            shutil.copy(file_path, join(imagestr, c + '_0000.nii.gz'))
    
    for c in patients_test:
        tr_files_2 = glob.glob(join(src_data_folder,c, "S*"))
        print(tr_files_2)
        shutil.copy(join(src_data_folder, c,  c + '.nii.gz'), join(labelstr, c + '.nii.gz'))
        for file_path in tr_files_2:
            shutil.copy(file_path, join(imagests, c + '_0000.nii.gz'))


    generate_dataset_json(
                          out_base, {0: "CT"},
                          labels = {
                              "background": 0,
                              "coadread": 1
                          },
                          file_ending=".nii.gz",
                          num_training_cases=num_training_cases,
                          )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_folder",
        type=str,
    )
    parser.add_argument(
        "-d", "--dataset_id", required=False, type=int, default=8, help="nnU-Net Dataset ID, default: 8"
    )
    args = parser.parse_args()
    print("Converting...")
    convert_COADREAD(args.input_folder, args.dataset_id)
    print("Done!")
