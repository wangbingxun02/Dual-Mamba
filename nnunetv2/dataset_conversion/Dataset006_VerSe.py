import os
from batchgenerators.utilities.file_and_folder_operations import *
import shutil

from nnunetv2.paths import nnUNet_raw
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

import random

random.seed(42)

def convert_VerSe(src_data_folder: str, dataset_id=6):
    

    task_name = "VerSe"

    foldername = "Dataset%03.0d_%s" % (dataset_id, task_name)

    directory = src_data_folder
    # setting up nnU-Net folders
    out_base = join(nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)

    # train&test
    train_suffix = "_ct.nii.gz"

    train_file_list = []

    abandon_file = ['sub-verse018_ct.nii.gz',
                    'sub-verse257_ct.nii.gz',
                    'sub-verse577_dir-ax_ct.nii.gz',
                    'sub-verse593_dir-sag_ct.nii.gz',
                    'sub-verse525_dir-sag_ct.nii.gz',
                    'sub-verse549_ct.nii.gz',
                    'sub-verse563_dir-iso_ct.nii.gz',
                    'sub-verse531_ct.nii.gz',
                    'sub-verse644_ct.nii.gz',
                    'sub-verse572_dir-sag_ct.nii.gz',
                    'sub-verse769_ct.nii.gz',
                    'sub-verse650_dir-iso_ct.nii.gz',
                    'sub-verse813_dir-sag_ct.nii.gz',
                    'sub-verse651_dir-iso_ct.nii.gz',
                    'sub-verse700_dir-sag_ct.nii.gz',
                    'sub-verse711_ct.nii.gz',
                    'sub-verse833_dir-ax_ct.nii.gz',
                    'sub-verse761_ct.nii.gz',
                    'sub-verse641_dir-ax_ct.nii.gz',
                    'sub-verse642_dir-sag_ct.nii.gz',
                    'sub-verse806_ct.nii.gz']

    # 使用os.walk()遍历目录及其子目录
    for root, dirs, files in os.walk(directory):
        # 遍历文件
        for file in files:
            # 判断文件名是否以指定后缀结尾
            if file.endswith(train_suffix) and file not in abandon_file:
                file = os.path.join(root, file)
                # 将文件路径添加到列表中
                train_file_list.append(file)

    # 打印结果
    # print(train_file_list)
    # print(len(train_file_list))

    patients_train = sorted(train_file_list)
    patients_test = random.sample(patients_train, 73)
    patients_train = [f for f in patients_train if f not in patients_test]
    num_training_cases = len(patients_train)

    # print(num_training_cases)

    #copy
    for c in patients_train:
        shutil.copy(c, join(imagestr, os.path.basename(c)[:-len("_ct.nii.gz")] + '_0000.nii.gz'))
    for c in patients_test:
        shutil.copy(c, join(imagests, os.path.basename(c)[:-len("_ct.nii.gz")] + '_0000.nii.gz'))



    # labels
    mask_suffix = "_seg-vert_msk.nii.gz"

    # 存储结果的列表
    label_file_list = []

    # 使用os.walk()遍历目录及其子目录
    for root, dirs, files in os.walk(directory):
        # 遍历文件
        for file in files:
            # 判断文件名是否以指定后缀结尾
            if file.endswith(mask_suffix):
                file = os.path.join(root, file)
                label_file_list.append(file)

    # 打印结果
    # print(label_file_list)
    # print(len(label_file_list))

    #copy
    for c in label_file_list:
        shutil.copy(c, join(labelstr, os.path.basename(c)[:-len("_seg-vert_msk.nii.gz")] + '.nii.gz'))

    generate_dataset_json(
                          out_base, {0: "CT"},
                          labels = {
                                "background":0,
                                "cervical spine C1":1,
                                "cervical spine C2":2,
                                "cervical spine C3":3,
                                "cervical spine C4":4,
                                "cervical spine C5":5,
                                "cervical spine C6":6,
                                "cervical spine C7":7,
                                "thoracic spine T1":8,
                                "thoracic spine T2":9,
                                "thoracic spine T3":10,
                                "thoracic spine T4":11,
                                "thoracic spine T5":12,
                                "thoracic spine T6":13,
                                "thoracic spine T7":14,
                                "thoracic spine T8":15,
                                "thoracic spine T9":16,
                                "thoracic spine T10":17,
                                "thoracic spine T11":18,
                                "thoracic spine T12":19,
                                "lumbar spine L1":20,
                                "lumbar spine L2":21,
                                "lumbar spine L3":22,
                                "lumbar spine L4":23,
                                "lumbar spine L5":24,
                                "lumbar spine L6":25,
                                "sacrum":26,
                                "cocygis":27,
                                "thoracic vertebra T13":28

                          },
                          file_ending=".nii.gz",
                          num_training_cases=num_training_cases,
                          license="CC-BY-SA 4.0",
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
        help="The downloaded VerSe dataset dir. ",
    )
    parser.add_argument(
        "-d", "--dataset_id", required=False, type=int, default=6, help="nnU-Net Dataset ID, default: 6"
    )
    args = parser.parse_args()
    print("Converting...")
    convert_VerSe(args.input_folder, args.dataset_id)
    print("Done!")
