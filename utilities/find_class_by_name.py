import importlib
import pkgutil

from batchgenerators.utilities.file_and_folder_operations import *


def recursive_find_python_class(folder: str, class_name: str, current_module: str):
    tr = None
    for importer, modname, ispkg in pkgutil.iter_modules([folder]):
        # print(modname, ispkg)
        if not ispkg:
            m = importlib.import_module(current_module + "." + modname)
            if hasattr(m, class_name):
                tr = getattr(m, class_name)
                break

    if tr is None:
        for importer, modname, ispkg in pkgutil.iter_modules([folder]):
            if ispkg:
                next_current_module = current_module + "." + modname
                tr = recursive_find_python_class(join(folder, modname), class_name, current_module=next_current_module)
            if tr is not None:
                break
    return tr

if __name__ == '__main__':
    nnunet_trainer = recursive_find_python_class(join('/staff/wangtiantong/nnU-Net/nnUNet_new/nnUnet_reconstructinon/nnunetv2', "training", "nnUNetTrainer"),
                                                'nnUNetTrainer_hrnet', 'nnunetv2.training.nnUNetTrainer')
    if nnunet_trainer is not None:
        print('找到了')
    else:
        print('没找到')