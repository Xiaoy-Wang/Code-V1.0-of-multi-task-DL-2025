import numpy as np
import torch
import os


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        # Saves model when validation loss decrease.
        # """
        if self.verbose:
            pass
            # self.trace_func(
            #     f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        save_dir = os.path.dirname(self.path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(model, self.path)
        # model_dir = 'E:\\PhD\\rehabilitation\\小论文\\论文3\\同济坐站角度相位预测'
        # if not os.path.exists(model_dir):
        #     os.makedirs(model_dir)  # 创建文件夹（如果不存在的话）

        # 然后再保存模型
        # torch.save(model.state_dict(), os.path.join(model_dir, 'model_1.pt'))
        # path = '/model_1.pt'
        # # os.makedirs(os.path.dirname(self.path), exist_ok=True)
        # # print('path',self.path)
        # # torch.save(model, self.path)
        # torch.save(model,path)
        # torch.save({'model': model.state_dict()}, 'model_name.pth')
        self.val_loss_min = val_loss



