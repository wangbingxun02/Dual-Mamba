from batchgenerators.utilities.file_and_folder_operations import *
import shutil

from nnunetv2.paths import nnUNet_raw
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

import random

random.seed(42)




def convert_HaN_Seg(src_data_folder: str, dataset_id=3):

    

    # Arbitrary task id. This is just to ensure each dataset ha a unique number. Set this to whatever ([0-999]) you
    # want
    
    task_name = "HaN_Seg"

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
    patients_test = random.sample(patients_train, 8)
    patients_train = [f for f in patients_train if f not in patients_test]
    num_training_cases = len(patients_train)

    for c in patients_train:
        shutil.copy(join(src_data_folder, c, 'Seg.nii.gz'), join(labelstr, c + '.nii.gz'))
        shutil.copy(join(src_data_folder, c, c + '_IMG_CT.nii.gz'), join(imagestr, c + '_0000.nii.gz'))
        shutil.copy(join(src_data_folder, c, c + '_IMG_MR_T1_Adjusted.nii.gz'), join(imagestr, c + '_0001.nii.gz'))
    
    for c in patients_test:
        shutil.copy(join(src_data_folder, c, 'Seg.nii.gz'), join(labelstr, c + '.nii.gz'))
        shutil.copy(join(src_data_folder, c, c + '_IMG_CT.nii.gz'), join(imagests, c + '_0000.nii.gz'))
        shutil.copy(join(src_data_folder, c, c + '_IMG_MR_T1_Adjusted.nii.gz'), join(imagests, c + '_0001.nii.gz'))

    generate_dataset_json(
                          out_base, {0: "CT",
                                     1: "MR_T1"},
                          labels = {
                                "background": 0,
                                "A_Carotid_L":1,
                                "A_Carotid_R":2,
                                "Arytenoid":3,
                                "Bone_Mandible":4,
                                "Brainstem":5,
                                "BuccalMucosa":6,
                                "Cavity_Oral":7,
                                "Cochlea_L":8,
                                "Cochlea_R":9,
                                "Cricopharyngeus":10,
                                "Esophagus_S":11,
                                "Eye_AL":12,
                                "Eye_AR":13,
                                "Eye_PL":14,
                                "Eye_PR":15,
                                "Glnd_Lacrimal_L":16,
                                "Glnd_Lacrimal_R":17,
                                "Glnd_Submand_L":18,
                                "Glnd_Submand_R":19,
                                "Glnd_Thyroid":20,
                                "Glottis":21,
                                "Larynx_SG":22,
                                "Lips":23,
                                "OpticChiasm":24,
                                "OpticNrv_L":25,
                                "OpticNrv_R":26,
                                "Parotid_L":27,
                                "Parotid_R":28,
                                "Pituitary":29,
                                "SpinalCord":30,

                          },
                          file_ending=".nii.gz",
                          num_training_cases=num_training_cases,
                          license="CC-BY-NC-ND 4.0",
                          description='',
                          reference='',
                          release='0')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_folder",
        type=str,
        help="The downloaded HaN-Seg dataset dir. ",
    )
    parser.add_argument(
        "-d", "--dataset_id", required=False, type=int, default=3, help="nnU-Net Dataset ID, default: 3"
    )
    args = parser.parse_args()
    print("Converting...")
    convert_HaN_Seg(args.input_folder, args.dataset_id)
    print("Done!")
