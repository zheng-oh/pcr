import torch
import torch.nn as nn
import sys
from torchvision import transforms,datasets


class Pred():
    def __init__(self):
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu" if torch.cuda.is_available() else "cuda")
        self.criterion = nn.CrossEntropyLoss()
        self.modelname = ''
        self.modelmsg = ''
        self.model = ''
        self.test_datasets = ''
        self.dataloaders = ''
        self.data_transforms64 = transforms.Compose(
            [
                transforms.Resize(
                    256,
                ),
                transforms.CenterCrop(224),
                transforms.Resize(64,),
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

    def run(self,modelname, dp):
        self.modelname = modelname
        self.get_model()
        self.preprocessing(dp)
        self.pred()

    def get_model(self):
        print("加载model")
        self.modelmsg = torch.load('./models/{}.pkl'.format(self.modelname))
        self.model = self.modelmsg["model"].to(self.device)
        # print(self.modelmsg)

    def pred(self):
        self.model.eval()
        total_loss = 0.0
        correct = 0.0
        totel_test = 0.0
        with torch.no_grad():
            for batch_id, (data, target) in enumerate(self.dataloaders):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                # print(output)
                total_loss += self.criterion(output, target).item()
                _, preds = torch.max(output, dim=1)
                print(output)
                print(preds)
                break
        #         correct += torch.sum(preds == target)
        #         totel_test +=len(data)
        #     accuracy = correct / totel_test
        #     print("Test_Loss:{:.4f}".format(accuracy))
        #     print("Test_Accuracy:{:.4f}".format(total_loss))


    def preprocessing(self, dp):
        if '18' in self.modelname:
            self.test_datasets = datasets.ImageFolder(dp, self.data_transforms64)
        else:
            self.test_datasets = datasets.ImageFolder(dp, self.data_transforms224)
        self.dataloaders = torch.utils.data.DataLoader(self.test_datasets, batch_size=8, shuffle=True)


def main():
    model_name = "agingregin50ie"
    dp = "./data/ie/agingregin"
    p = Pred()
    p.run(model_name, dp)

if __name__ == '__main__':
    main()
    print("结束")