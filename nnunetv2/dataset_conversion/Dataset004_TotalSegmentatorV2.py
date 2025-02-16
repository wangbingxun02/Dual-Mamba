from batchgenerators.utilities.file_and_folder_operations import *
import shutil

from nnunetv2.paths import nnUNet_raw
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

import random

random.seed(42)




def convert_TotalSegmentatorV2(src_data_folder: str, dataset_id=4):

    

    # Arbitrary task id. This is just to ensure each dataset ha a unique number. Set this to whatever ([0-999]) you
    # want
    
    task_name = "TotalSegmentatorV2"

    foldername = "Dataset%03.0d_%s" % (dataset_id, task_name)

    # setting up nnU-Net folders
    out_base = join(nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)

    case_ids = subdirs(src_data_folder, prefix='s', join=False)
    patients_train = sorted(case_ids)
    patients_test = random.sample(patients_train, 248)
    patients_train = [f for f in patients_train if f not in patients_test]
    num_training_cases = len(patients_train)

    for c in patients_train:
        shutil.copy(join(src_data_folder, c, 'seg.nii.gz'), join(labelstr, c + '.nii.gz'))
        shutil.copy(join(src_data_folder, c, 'ct.nii.gz'), join(imagestr, c + '_0000.nii.gz'))
    
    for c in patients_test:
        shutil.copy(join(src_data_folder, c, 'seg.nii.gz'), join(labelstr, c + '.nii.gz'))
        shutil.copy(join(src_data_folder, c, 'ct.nii.gz'), join(imagests, c + '_0000.nii.gz'))


    generate_dataset_json(
                          out_base, {0: "CT"},
                          labels = {
                                "background": 0,
                                "spleen":1,
                                "kidney_right":2,
                                "kidney_left":3,
                                "gallbladder":4,
                                "liver":5,
                                "stomach":6,
                                "pancreas":7,
                                "adrenal_gland_right":8,
                                "adrenal_gland_left":9,
                                "lung_upper_lobe_left":10,
                                "lung_lower_lobe_left":11,
                                "lung_upper_lobe_right":12,
                                "lung_middle_lobe_right":13,
                                "lung_lower_lobe_right":14,
                                "esophagus":15,
                                "trachea":16,
                                "thyroid_gland":17,
                                "small_bowel":18,
                                "duodenum":19,
                                "colon":20,
                                "urinary_bladder":21,
                                "prostate":22,
                                "kidney_cyst_left":23,
                                "kidney_cyst_right":24,
                                "sacrum":25,
                                "vertebrae_S1":26,
                                "vertebrae_L5":27,
                                "vertebrae_L4":28,
                                "vertebrae_L3":29,
                                "vertebrae_L2":30,
                                "vertebrae_L1":31,
                                "vertebrae_T12":32,
                                "vertebrae_T11":33,
                                "vertebrae_T10":34,
                                "vertebrae_T9":35,
                                "vertebrae_T8":36,
                                "vertebrae_T7":37,
                                "vertebrae_T6":38,
                                "vertebrae_T5":39,
                                "vertebrae_T4":40,
                                "vertebrae_T3":41,
                                "vertebrae_T2":42,
                                "vertebrae_T1":43,
                                "vertebrae_C7":44,
                                "vertebrae_C6":45,
                                "vertebrae_C5":46,
                                "vertebrae_C4":47,
                                "vertebrae_C3":48,
                                "vertebrae_C2":49,
                                "vertebrae_C1":50,
                                "heart":51,
                                "aorta":52,
                                "pulmonary_vein":53,
                                "brachiocephalic_trunk":54,
                                "subclavian_artery_right":55,
                                "subclavian_artery_left":56,
                                "common_carotid_artery_right":57,
                                "common_carotid_artery_left":58,
                                "brachiocephalic_vein_left":59,
                                "brachiocephalic_vein_right":60,
                                "atrial_appendage_left":61,
                                "superior_vena_cava":62,
                                "inferior_vena_cava":63,
                                "portal_vein_and_splenic_vein":64,
                                "iliac_artery_left":65,
                                "iliac_artery_right":66,
                                "iliac_vena_left":67,
                                "iliac_vena_right":68,
                                "humerus_left":69,
                                "humerus_right":70,
                                "scapula_left":71,
                                "scapula_right":72,
                                "clavicula_left":73,
                                "clavicula_right":74,
                                "femur_left":75,
                                "femur_right":76,
                                "hip_left":77,
                                "hip_right":78,
                                "spinal_cord":79,
                                "gluteus_maximus_left":80,
                                "gluteus_maximus_right":81,
                                "gluteus_medius_left":82,
                                "gluteus_medius_right":83,
                                "gluteus_minimus_left":84,
                                "gluteus_minimus_right":85,
                                "autochthon_left":86,
                                "autochthon_right":87,
                                "iliopsoas_left":88,
                                "iliopsoas_right":89,
                                "brain":90,
                                "skull":91,
                                "rib_right_4":92,
                                "rib_right_3":93,
                                "rib_left_1":94,
                                "rib_left_2":95,
                                "rib_left_3":96,
                                "rib_left_4":97,
                                "rib_left_5":98,
                                "rib_left_6":99,
                                "rib_left_7":100,
                                "rib_left_8":101,
                                "rib_left_9":102,
                                "rib_left_10":103,
                                "rib_left_11":104,
                                "rib_left_12":105,
                                "rib_right_1":106,
                                "rib_right_2":107,
                                "rib_right_5":108,
                                "rib_right_6":109,
                                "rib_right_7":110,
                                "rib_right_8":111,
                                "rib_right_9":112,
                                "rib_right_10":113,
                                "rib_right_11":114,
                                "rib_right_12":115,
                                "sternum":116,
                                "costal_cartilages":117

                          },
                          file_ending=".nii.gz",
                          num_training_cases=num_training_cases,
                          license="CC-BY 4.0",
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
        help="The downloaded TotalSegmentatorV2 dataset dir. ",
    )
    parser.add_argument(
        "-d", "--dataset_id", required=False, type=int, default=4, help="nnU-Net Dataset ID, default: 4"
    )
    args = parser.parse_args()
    print("Converting...")
    convert_TotalSegmentatorV2(args.input_folder, args.dataset_id)
    print("Done!")
