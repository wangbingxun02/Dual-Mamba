## Dual-Mamba: A Hybrid CNN-Mamba Architecture for Tumor Segmentation from 3D Medical Images

### Installation
requirements:
- python >= 3.11
- torch >= 2.0.0
- CUDA >= 12.4
- Ubuntu >= 22.04.5  
We recommend using the above version to avoid unknown bugs.

```bash
git clone https://github.com/wangbingxun02/Dual-Mamba.git
cd Dual-Mamba
pip install -e .
```
We recommend that you use a virtual environment, and the above steps automatically install the necessary dependency libraries, especially mamba and torch.

### preprocessing
First, consult [nnUNet's official documentation](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md) to prepare and transform the dataset

Then, you can start preprocessing by running the following command:
```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

### training
Start training a Dual-Mamba model by running the following command:
```bash
nnUNetv2_train DATASET_ID 3d_fullres 0(use 0-4 if you want to use 5-fold cross-validation) --model dual_mamba
```
We have provided the code for the other models in the paper. If you want to verify or compare the performance of other models with Dual-Mamba, just modify the --model parameter, for example:  
```bash
nnUNetv2_train DATASET_ID 3d_fullres 0(use 0-4 if you want to use 5-fold cross-validation) --model CoTr
```
### inference
If you want to use 5-fold cross-validation and have trained 5-fold model, then run the following command:
```bash
nnUNetv2_predict -i INPUT_DIR -o OUTPUT_DIR -d DATASET_ID -c 3d_fullres -tr nnUNetTrainer_dual_mamba 
```
Or if you want to inference a single fold, then run the following command:
```bash
nnUNetv2_predict -i INPUT_DIR -o OUTPUT_DIR -d DATASET_ID -c 3d_fullres -tr nnUNetTrainer_dual_mamba -f (0-4)
```
