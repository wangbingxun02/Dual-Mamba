#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from collections import OrderedDict
from nnunetv2.paths import nnUNet_raw
from batchgenerators.utilities.file_and_folder_operations import *
import shutil
import SimpleITK as sitk
import random

random.seed(42)

def convert_for_submission(source_dir, target_dir):
    """
    I believe they want .nii, not .nii.gz
    :param source_dir:
    :param target_dir:
    :return:
    """
    files = subfiles(source_dir, suffix=".nii.gz", join=False)
    maybe_mkdir_p(target_dir)
    for f in files:
        img = sitk.ReadImage(join(source_dir, f))
        out_file = join(target_dir, f[:-7] + ".nii")
        sitk.WriteImage(img, out_file)



if __name__ == "__main__":
    base = "/staff/wangbingxun/dataset/SegTHOR"

    task_id = 59
    task_name = "SegTHOR"

    foldername = "Dataset%03.0d_%s" % (task_id, task_name)

    out_base = join(nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)

    train_patient_names = []
    test_patient_names = []
    train_patients = subfolders(join(base, "train"), join=False)
    for p in train_patients:
        curr = join(base, "train", p)
        label_file = join(curr, "GT.nii.gz")
        image_file = join(curr, p + ".nii.gz")
        shutil.copy(image_file, join(imagestr, p + "_0000.nii.gz"))
        shutil.copy(label_file, join(labelstr, p + ".nii.gz"))
        train_patient_names.append(p)

    # 从训练集中划分测试集
    
    file_list = sorted(os.listdir(imagestr))
    selected_files = random.sample(file_list, 10)
    for file in selected_files:
        file_path = os.path.join(imagestr, file)
        new_path = os.path.join(imagests, file)
        shutil.move(file_path, new_path)
    
    file_list= [file for file in os.listdir(imagestr) ]

    # test_patients = subfiles(join(base, "test"), join=False, suffix=".nii.gz")
    # for p in test_patients:
    #     p = p[:-7]
    #     curr = join(base, "test")
    #     image_file = join(curr, p + ".nii.gz")
    #     shutil.copy(image_file, join(imagests, p + "_0000.nii.gz"))
    #     test_patient_names.append(p)

    from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

    generate_dataset_json(
        out_base, {0: "CT"},
        labels={
            "background": 0,
            "esophagus": 1,
            "heart": 2,
            "trachea": 3,
            "aorta": 4
        },
        description="SegTHOR",
        file_ending=".nii.gz",
        num_training_cases=len(file_list),
        dataset_name="SegTHOR",
        reference="see challenge website",
        release="0.0",
        license="see challenge website"
    )

    # json_dict = OrderedDict()
    # json_dict['name'] = "SegTHOR"
    # json_dict['description'] = "SegTHOR"
    # json_dict['tensorImageSize'] = "4D"
    # json_dict['reference'] = "see challenge website"
    # json_dict['licence'] = "see challenge website"
    # json_dict['release'] = "0.0"
    # json_dict['labels'] = {
    #     "background": 0,
    #     "esophagus": 1,
    #     "heart": 2,
    #     "trachea": 3,
    #     "aorta": 4
    # }
    # json_dict['numTraining'] = len(train_patient_names)
    # json_dict['numTest'] = len(test_patient_names)
    # json_dict['file_ending'] = ".nii.gz"
    # json_dict['channel_names'] = {
    #     "0": "CT"
    # }
    
    
    # # json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i.split("/")[-1], "label": "./labelsTr/%s.nii.gz" % i.split("/")[-1]} for i in
    # #                          train_patient_names]
    # # json_dict['test'] = ["./imagesTs/%s.nii.gz" % i.split("/")[-1] for i in test_patient_names]

    # save_json(json_dict, os.path.join(out_base, "dataset.json"))
