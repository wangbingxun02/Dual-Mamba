



import torch






""" def my_get_dice_loss(preds: torch.tensor, target):
    pred = torch.softmax(preds, dim=1)[:, 1, :, :].unsqueeze(1)
    if isinstance(target, list):  # 如果 target 是一个列表
        print(111111111111111)
        #target = torch.tensor(target).float()  # 将 target 转换为浮点数张量
    elif isinstance(target, torch.Tensor):  # 如果 target 是一个张量
        print(222222222222222222)
        #target = target.float()  # 将 target 转换为浮点数张量
    inter = (pred * target)
    union = (pred + target)
    # pred:BCHW, target: BCHW
    if (len(pred.shape) == 4) and (len(target.shape) == 4):
        inter = inter.sum(dim=1).sum(dim=1).sum(dim=1)
        union = union.sum(dim=1).sum(dim=1).sum(dim=1)
    dice_loss = 1 - 2 * (inter + 1) / (union + 2)
    return dice_loss.mean() """

def my_get_dice_loss(preds: torch.tensor, target: torch.tensor):
    pred=torch.softmax(preds,dim=1)[:,1,:,:].unsqueeze(1)
    #print(pred.shape)
    inter = (pred * target)
    union = (pred + target)
    # pred:BCHW, target: BCHW
    if (len(pred.shape) == 4) and (len(target.shape) == 4):
        #print("yes")
        inter = inter.sum(dim=1).sum(dim=1).sum(dim=1)
        union = union.sum(dim=1).sum(dim=1).sum(dim=1)
    dice_loss = 1 - 2 * (inter + 1) / (union + 2)
    #print(dice_loss.mean())
    return dice_loss.mean()