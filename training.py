import torch
from torch import nn
from torchvision import models
import logging
import preprocessing
import time


def get_dataloaders(dp,batch_size):
    datasets = {}
    image_datasets, image_dataloaders = preprocessing.run(dp, net_num=net_num)
    samp_num = len(image_datasets)
    train_num, vaild_num = int(0.6 * samp_num), samp_num - int(0.6 * samp_num)
    datasets["train"], datasets["valid"] = torch.utils.data.random_split(image_datasets, [train_num,vaild_num])
    dataloaders = {
        x: torch.utils.data.DataLoader(
            dataset=datasets[x], batch_size=batch_size, shuffle=True
        )
        for x in ["train", "valid"]
    }
    return dataloaders, image_datasets.classes, train_num, vaild_num


def get_model(class_num):
    if net_num == 50:
        model_pre = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    elif net_num == 18:
        model_pre = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model_pre.parameters():
        param.requires_grad = False
    print(model_pre.fc.in_features)
    model_pre.fc = nn.Sequential(
        nn.Linear(model_pre.fc.in_features, len(class_num)),
        nn.LogSoftmax(dim=1),
    )
    return model_pre


def train(model, device, train_loader, criterion, optimizer, scheduler, train_num):
    model.train()
    model = model.to(device)
    correct = 0.0
    total_loss = 0.0  # 初始化
    for batch_id, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss
        _, preds = torch.max(output, dim=1)
        correct += torch.sum(preds == target)
    accuracy = correct / train_num
    scheduler.step()
    return accuracy, total_loss


def vaild(model, device, valid_loader, criterion, vaild_num):
    model.eval()
    correct = 0.0
    total_loss = 0.0
    with torch.no_grad():
        for batch_id, (data, target) in enumerate(valid_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target).item()
            _, preds = torch.max(output, dim=1)
            correct += torch.sum(preds == target)
        total_loss += loss
        accuracy = correct / vaild_num
        return accuracy, total_loss


def run(model, dataloaders, num_epochs, train_num, vaild_num):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optim_fit = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optim_fit, step_size=10, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    global best_model
    best_model = {"model": '', "train_acc": 0, "train_loss": 0, "valid_acc": 0, "valid_loss": 0}
    for epoch in range(num_epochs):
        logging.info("训练迭代：%d" % epoch)
        train_acc, train_loss = train(model, device, dataloaders["train"], criterion, optim_fit, scheduler, train_num)
        valid_acc, valid_loss = vaild(model, device, dataloaders["valid"], criterion, vaild_num)
        if train_acc+valid_acc > best_model["train_acc"]+best_model["valid_acc"]:
            best_model["train_acc"] = train_acc
            best_model["valid_acc"] = valid_acc
            best_model["train_loss"] = train_loss
            best_model["valid_loss"] = valid_loss
            best_model["model"] = model
            logging.info("{} 最优模型:train_acc:{},train_loss:{},valid_acc:{},valid_loss:{}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), best_model["train_acc"],best_model["train_loss"],best_model["valid_acc"],best_model["valid_loss"]))
            torch.save(best_model, "./models/{0}{1}{2}.pkl".format(aim,net_num,cate))


def main():
    global net_num, cate, aim
    net_num = 50
    cate = "oe"
    aim = "agingregin"
    num_epochs = 200
    dp = "./data/{}/agingregin".format(cate)
    logging.basicConfig(filename='./logs/train_{0}_{1}_{2}.log'.format(aim, cate, net_num), level=logging.INFO)
    dataloaders, class_num, train_num, vaild_num = get_dataloaders(dp, 8)
    model = get_model(class_num)
    logging.info('{} Started'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    run(model, dataloaders, num_epochs, train_num, vaild_num)

if __name__ == '__main__':
    main()
    logging.info('{} Finished'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

