import os.path
# import sys
# sys.path.append("..")
import torch
import torch.nn as nn
import preprocessing
from torchvision import transforms
from result import Resultplt
import numpy as np


class Test:
    def __init__(self):
        self.device = torch.device("cpu" if torch.cuda.is_available() else "cuda")
        self.criterion = nn.CrossEntropyLoss()
        self.modelname = ''
        self.modelmsg = ''
        self.model = ''
        self.net_num = 50
        self.num_classes = 0
        self.test_datasets = ''
        self.dataloaders = ''
        self.test_num = 0
        self.lables = []  # 存储标签名称
        self.target_list = []  # 存储真实标签
        self.score_list = []  # 存储预测得分
        self.preds_list = []  # 存储预测标签
        self.matrix = []  # 存储混淆矩阵
        self.test_loss = 0
        self.test_acc = 0
        self.data_transforms64 = transforms.Compose(
            [
                transforms.Resize(
                    256,
                ),
                transforms.CenterCrop(224),
                transforms.Resize(64, ),
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.data_transforms224 = transforms.Compose(
            [
                transforms.Resize(
                    256,
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def get_model(self):
        print("加载model：{}".format(self.modelname))
        self.modelmsg = torch.load('./models/{}.pkl'.format(self.modelname))
        self.model = self.modelmsg["model"].to(self.device)
        # print(self.modelmsg)

    def get_dataloaders(self, dp, batch_size):
        t_datasets, self.dataloaders = preprocessing.run(dp, stage="test", net_num=self.net_num, batch_size=batch_size)
        self.test_num = len(t_datasets)
        return t_datasets.classes

    def confusion_matrix(self, conf_matrix):
        # print(self.preds_list)
        # print(self.target_list)
        for p, t in zip(self.preds_list, self.target_list):
            conf_matrix[p, t] += 1
        return conf_matrix

    def pred(self):
        self.model.eval()
        correct = 0.0
        total_loss = 0.0
        self.matrix = torch.zeros(self.num_classes, self.num_classes)  # 创建一个空矩阵存储混淆矩阵
        with torch.no_grad():
            for batch_id, (data, target) in enumerate(self.dataloaders):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target).item()
                e_class, preds = torch.max(output, dim=1)
                print(output)
                print(output.shape)
                print(e_class)
                print(preds)
                correct += torch.sum(preds == target)
                self.score_list.extend(output.detach().cpu().numpy())
                self.target_list.extend(target.cpu().numpy())
                self.preds_list.extend(preds.cpu().numpy())
                break
            self.test_loss += loss
            self.test_acc = correct / self.test_num
            self.matrix = self.confusion_matrix(self.matrix)
            self.matrix = self.matrix.cpu()
            self.matrix = np.array(self.matrix)
            # corrects = self.matrix.diagonal(offset=0)
            # per_kinds = self.matrix.sum(axis=1)
            # print("混淆矩阵总元素个数：{0},测试集总个数:{1}".format(int(np.size(self.matrix)), self.test_num))
            # 获取每种Emotion的识别准确率
            # print("每种类别总个数：", per_kinds)
            # print("每种类别预测正确的个数：", corrects)
            # print("每种类别的识别准确率为：{0}".format([rate * 100 for rate in corrects / per_kinds]))

    def plt(self):
        p = Resultplt(self.num_classes, self.target_list, self.score_list, self.matrix, self.lables, self.test_acc)
        # p.roc("roc_{}".format(self.modelname))
        # p.conf_mat("matrix_{}".format(self.modelname))
        p.max_roc("max_roc_{}".format(self.modelname))

    def run(self, modelname, dp, net_num):
        self.modelname = modelname
        self.get_model()
        self.net_num = net_num
        classes = self.get_dataloaders(dp, 8)
        self.lables = classes
        self.num_classes = len(classes)
        print(classes)
        self.pred()
        result = {"test_acc": self.test_acc, "test_loss": self.test_loss}
        print("test_acc:{:.2f},test_loss{:.2f}".format(result["test_acc"], result["test_loss"]))
        return
        if not os.path.exists("results"):
            os.mkdir("results")
        torch.save(result, "./results/{}.pkl".format(modelname))
        self.plt()


def main():
    # models = ["CK18", "CK50", "DZ18", "DZ50", "NT18", "NT50", "SJ18", "SJ50", "SS18", "SS50"]
    # models = ["agingregin50ie", "agingregin18ie"]
    models = ["agingregin18ie"]
    for model_name in models:
        if "agingregin" in model_name:
            dp = "./data/test/agingreginIE"
        else:
            dp = "./data/test/agingIE/" + model_name[:2]
        net_num = 18 if "18" in model_name else 50
        t = Test()
        t.run(model_name, dp, net_num)


if __name__ == '__main__':
    main()
    print("结束")
