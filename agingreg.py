import os.path

import torch
from torchvision import transforms, datasets, models
import torchvision
import torch.nn as nn
from torchvision.utils import make_grid
import matplotlib
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import numpy as np
from torchvision.io import read_image
from torch.utils.tensorboard import SummaryWriter

# matplotlib.use("TkAgg")
def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(img)
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()
    
def get_model(outfeature):
    model_pre = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1a)
    # model_pre = models.resnet50(pretrained=True)
    for param in model_pre.parameters():
        param.requires_grad = False
    num_ftrs = model_pre.fc.in_features
    print(num_ftrs)
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_pre.fc = nn.Linear(num_ftrs, outfeature)
    return model_pre

def train(model, device, train_loader, criterion, optimizer, epoch, writer,name):
    model.train()
    model = model.to(device)
    total_loss = 0.0  # 初始化
    for batch_id, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss
    writer.add_scalar(name+"--Train loss", total_loss / len(train_loader), epoch)
    writer.flush()
    return total_loss / len(train_loader)

def test(model, device, test_loader, criterion, epoch, writer,name):
    model.eval()
    total_loss = 0.0
    correct = 0.0
    totel_test = 0.0
    best_model = ''
    best_accuract = 0.0
    best_epoch = 0
    with torch.no_grad():
        for batch_id, (data, target) in enumerate(test_loader):
        # for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            _, preds = torch.max(output, dim=1)
            correct += torch.sum(preds == target)
            totel_test += len(target)
        # total_loss /= totel_test
        accuracy = correct / totel_test
        writer.add_scalar(name+"Test loss", total_loss, epoch)
        writer.add_scalar(name+"Accuracy", accuracy, epoch)
        writer.flush()
        print("Test Loss:{:.4f},Accuracy:{:.4f}".format(total_loss, accuracy))
        if accuracy>best_accuract:
            best_model = model
            best_accuract = accuracy
            best_epoch = epoch


def main(num_epochs,batch_size,dt):
    writer = SummaryWriter("./logs/"+dt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print(type(device))
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.Resize(
                    250,
                ),
                transforms.RandomRotation(30),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "valid": transforms.Compose(
            [
                transforms.Resize(250),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }
    data_path = os.path.join("./data/oe",dt)
    print(data_path)
    # image_datasets = {x: datasets.ImageFolder(os.path.join(data_path, x), data_transforms[x])
    #     for x in ["train", "valid"]
    # }
    # dataloaders = {
    #     x: torch.utils.data.DataLoader(
    #         dataset=image_datasets[x], batch_size=batch_size, shuffle=True
    #     )
    #     for x in ["train", "valid"]
    # }
    # data_size = {x: len(image_datasets[x]) for x in ["train", "valid"]}
    all_datasets = datasets.ImageFolder(data_path,data_transforms['train'])
    print(len(all_datasets))
    print(all_datasets.classes)
    outfeatures = len(all_datasets.classes)
    print(outfeatures)
    train_num,valid_num = 160,30
    print("训练数据:%d,验证数据：%d"%(train_num,valid_num))
    image_datasets={}
    image_datasets["train"], image_datasets["valid"] = torch.utils.data.random_split(all_datasets, [train_num,valid_num])
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True)
        for x in ["train", "valid"]
    }
    data_size = {x: len(image_datasets[x]) for x in ["train", "valid"]}
    images, targets = next(iter(dataloaders["train"]))
    print(targets)
    writer.add_images("chenpi", images)
    writer.flush()
    # out = make_grid(images, nrow=4, padding=10)
    # show(out)
    model = get_model(outfeatures)
    # 优化器
    optim_fit = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optim_fit, step_size=10, gamma=0.1)
    criterion = nn.NLLLoss()
    # print(len(dataloaders['train']))

    print(len(dataloaders['valid']))
    name = dt+str(num_epochs)
    for epoch in range(num_epochs):
        print("训练迭代：%d" % epoch)
        train(model, device, dataloaders["train"], criterion, optim_fit, epoch, writer,name)
        test(model, device, dataloaders["valid"], criterion, epoch, writer,name)

if __name__ == "__main__":
    # dt = "regin"
    dt = "agingregin"
    # dt = "aging"
    num_epochs = 800
    batch_size = 8
    main(num_epochs,batch_size,dt)