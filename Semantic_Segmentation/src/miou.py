import numpy as np


class IOUMetric:
    """
     Class to calculate mean-iou using fast_hist method
     """
    def __init__(self, num_class):
        self.num_class = num_class
        # hist is a confusion matrix of all batches
        self.hist = np.zeros((num_class, num_class))

    def _fast_hist(self, label_pred, label_true):
        """produce a hist 

        Args:
            label_pred (ndarray): predicted labels of single dimension
            label_true (ndarray): ground truth of single dimension

        Returns:
            ndarray: a confusion matrix, of shape (self.num_class, self.num_class)
        """
        n = self.num_class
        mask = (label_true >= 0) & (label_true < n)
        hist = np.bincount(n * label_true[mask].astype(int) + label_pred[mask],
                           minlength=n**2).reshape(n, n)
        return hist

    def add_batch(self, label_pred, label_true):
        """add into self.hist in btches

        Args:
            label_pred (ndarray): of shape (batch_size, H, W)
            label_true (ndarray): of shape (batch_size, H, W)
        """
        self.hist += self._fast_hist(label_pred.flatten(), label_true.flatten())

    def evaluate(self):
        """compute the acc (accuracy of all pixels), mean accuracy of all classes (acc_cls), iou of each class and mean-iou of all classes

        Returns:
            dict: a dict that contains acc, acc_cls, iou and miou
        """
        # acc: float
        acc = np.diag(self.hist).sum() / self.hist.sum()

        # acc_cls: float
        acc_cls = np.diag(self.hist) / self.hist.sum(axis=1)
        # acc_cls = np.nanmean(acc_cls)

        # iou: ndarray
        iou = np.diag(self.hist) / (self.hist.sum(axis=1) +
                                    self.hist.sum(axis=0) - np.diag(self.hist))

        # miou: float
        miou = np.nanmean(iou)

        # ! what is freq and fwvacc?
        # freq = self.hist.sum(axis=1) / self.hist.sum()
        # fwavacc = (freq[freq > 0] * iou[freq > 0]).sum()

        return {'acc': acc, 'acc_cls': acc_cls, 'iou': iou, 'miou': miou}
