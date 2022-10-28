import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from itertools import cycle
import torch



class Resultplt:

    def __init__(self, n_classes, label_list, score_list, matrix, lables, test_acc):
        self.n_classes = n_classes
        self.label_list = label_list
        self.score_list = score_list
        self.matrix = matrix
        self.lables = lables
        self.test_acc = test_acc

    def conf_mat(self, name):
            # def confusion_matrix(preds, labels, conf_matrix):
        plt.figure(figsize=(16, 10))

        plt.imshow(self.matrix, cmap=plt.cm.Blues)
        # 设置x轴坐标label
        plt.xticks([i for i in range(self.n_classes)], self.lables, rotation=45)
        # 设置y轴坐标label
        plt.yticks([i for i in range(self.n_classes)], self.lables)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        # plt.title('Confusion matrix (acc='+self.summary()+')')
        plt.title('Confusion matrix (acc={:.2f})'.format(self.test_acc))

        # 在图中标注数量/概率信息
        thresh = self.matrix.max() / 2
        for x in range(self.n_classes):
            for y in range(self.n_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(self.matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.savefig('./results/{}.pdf'.format(name))

    def roc(self, name):
        score_array = np.array(self.score_list)
        # 将label转换成onehot形式
        label_tensor = torch.tensor(self.label_list)
        label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
        label_onehot = torch.zeros(label_tensor.shape[0], self.n_classes)
        label_onehot.scatter_(dim=1, index=label_tensor, value=1)
        label_onehot = np.array(label_onehot)
        lw = 2
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(self.n_classes):
            fpr[i], tpr[i], _ = roc_curve(label_onehot[:,i], score_array[:,i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(self.n_classes)]))
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        fpr_dict = dict()
        tpr_dict = dict()
        roc_auc_dict = dict()
        for i in range(self.n_classes):
            fpr_dict[i], tpr_dict[i], _ = roc_curve(label_onehot[:, i], score_array[:, i])
            roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
        # micro
        fpr_dict["micro"], tpr_dict["micro"], _ = roc_curve(label_onehot.ravel(), score_array.ravel())
        roc_auc_dict["micro"] = auc(fpr_dict["micro"], tpr_dict["micro"])

        # macro
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(self.n_classes)]))
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(self.n_classes):
            mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
        # Finally average it and compute AUC
        mean_tpr /= self.n_classes
        fpr_dict["macro"] = all_fpr
        tpr_dict["macro"] = mean_tpr
        roc_auc_dict["macro"] = auc(fpr_dict["macro"], tpr_dict["macro"])

        # 绘制所有类别平均的roc曲线
        plt.figure(figsize=(16, 10))
        lw = 2
        plt.plot(fpr_dict["micro"], tpr_dict["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                 ''.format(roc_auc_dict["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr_dict["macro"], tpr_dict["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                 ''.format(roc_auc_dict["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(self.n_classes), colors):
            plt.plot(fpr_dict[i], tpr_dict[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(i, roc_auc_dict[i]))
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        plt.savefig('./results/{}.pdf'.format(name))



class Resultexport:

    def __init__(self):
        pass

