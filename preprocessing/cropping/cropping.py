import numpy as np


# Hello! crop_to_nonzero is the function you are looking for. Ignore the rest.
from acvl_utils.cropping_and_padding.bounding_boxes import get_bbox_from_mask, crop_to_bbox, bounding_box_to_slice

'''

四维数组array(CXYZ)中，第一个维度的最后array[-1,:,:,:]存储的是分割标注结果。而第一个维度的前面存储不同模态的数据，
如MRI数据中有FLAIR, T1w, t1gd, T2w等四种模态，array[0,:,:,:]表示FLAIR序列成像的强度数据，array[1,:,:,:]表示T1加权的强度数据，
以此类推。如果仅单模态，则四维数组第一维度长度仅为2，分别表示影像数据以及标注数据。
四维数组array的后三个维度代表x,y,z三个坐标表示的三维数据，对于原始影像数据，值大小代表强度，而对于标注结果，后三个维度的三维数据值分别为0，1，2……表示不同的标注类别。
在后续的代码中，为了简便，将不同模态的原始图像与分割标注分开，使用data(CXYZ)代表四维图像数据，使用seg(XYZ)代表三维标注数据。


'''
def create_nonzero_mask(data):
    """

    :param data:
    :return: the mask is True where the data is nonzero  创建非零区域的掩码，该掩码与数据具有相同的形状，其中非零像素对应的位置为True，零像素对应的位置为False。
    """
    from scipy.ndimage import binary_fill_holes
    assert data.ndim in (3, 4), "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
    nonzero_mask = np.zeros(data.shape[1:], dtype=bool)
    for c in range(data.shape[0]):
        this_mask = data[c] != 0
        nonzero_mask = nonzero_mask | this_mask
    nonzero_mask = binary_fill_holes(nonzero_mask)
    return nonzero_mask


def crop_to_nonzero(data, seg=None, nonzero_label=-1):
    """

    :param data:
    :param seg:
    :param nonzero_label: this will be written into the segmentation map
    :return:
    """
    nonzero_mask = create_nonzero_mask(data)
    bbox = get_bbox_from_mask(nonzero_mask)

    slicer = bounding_box_to_slice(bbox)
    data = data[tuple([slice(None), *slicer])]

    if seg is not None:
        seg = seg[tuple([slice(None), *slicer])]

    nonzero_mask = nonzero_mask[slicer][None]
    if seg is not None:
        # 将背景像素值定位nonzero_label
        seg[(seg == 0) & (~nonzero_mask)] = nonzero_label
    # 没有先前的分割结果时，创建一个新的分割图
    else:
        nonzero_mask = nonzero_mask.astype(np.int8)
        # 将背景像素值定位nonzero_label
        nonzero_mask[nonzero_mask == 0] = nonzero_label
        # 将非零区域像素标记为0
        nonzero_mask[nonzero_mask > 0] = 0
        seg = nonzero_mask
    return data, seg, bbox


