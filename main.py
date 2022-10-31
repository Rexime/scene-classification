import os
import time
import glob
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from PIL import Image
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets, utils
from alexnet import AlexNet, AlexNet_BN
from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18, resnet50


class SceneDataset(Dataset):
    def __init__(self, annotations_csv, root_dir, transform=None):
        self.annotations = pd.read_csv(annotations_csv)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        label = torch.tensor(int(self.annotations.iloc[index, 1]))
        if self.transform:
            image = self.transform(image)
        return [image, label]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

data_transform = transforms.Compose([transforms.Resize((256, 256)),
                                    transforms.CenterCrop((224, 224)),
                                     transforms.RandomVerticalFlip(),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

# # 1.AlexNet
# net = AlexNet()

# 2.ResNet18
net = resnet18(pretrained=True)
net.fc = nn.Linear(net.fc.in_features, 100)

# # 3.ResNet50
# net = resnet50(pretrained=True)
# net.fc = nn.Linear(net.fc.in_features, 100)

# # 4.EfficientNet
# net = EfficientNet.from_pretrained('efficientnet-b0')   # b3
# net._fc = nn.Linear(net._fc.in_features, 100)


def train():
    LEARNING_RATE = 0.001
    BATCH_SIZE = 128
    EPOCHS = 30

    dataset = SceneDataset(annotations_csv="/home/ubuntu/MONY/1/cvdl2021-scene-classification/train_labels.csv",
                           root_dir="/home/ubuntu/MONY/1/cvdl2021-scene-classification/train/train",
                           transform=data_transform)

    n_val = int(len(dataset) * 0.2)       # 20% 验证集
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    net.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    save_path = './weights/resnet50.pth'
    best_acc = 0.0
    for epoch in range(EPOCHS):
        # train
        net.train()
        running_loss = 0.0
        for step, (data, labels) in enumerate(train_loader):

            outputs = net(data.to(device))
            loss = loss_function(outputs, labels.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            rate = (step + 1) / len(train_loader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")

        # validate
        net.eval()
        acc = 0.0  
        with torch.no_grad():
            for val_data in val_loader:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += (predict_y == val_labels.to(device)).sum().item()
            val_accurate = acc / len(val_dataset)
            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(net.state_dict(), save_path)
            print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
                  (epoch + 1, running_loss / step, val_accurate))


def test():
    net.load_state_dict(torch.load('./resnet18_0.0001.pth'))
    net.to(device)
    net.eval()

    test_image_path = "./test/test/"
    test_image_files = glob.glob(test_image_path + '*')
    image_name_list = []
    predict_res_list = []
    for image in test_image_files:
        image_name = os.path.basename(image)
        image_name_list.append(image_name)
        img = Image.open(image)
        img = data_transform(img)
        img = torch.unsqueeze(img, dim=0)
        with torch.no_grad():
            output = net(img.to(device))
            output = torch.squeeze(output)
            predict = torch.softmax(output, dim=0).detach().cpu()
            predict_class = torch.argmax(predict).numpy()
            predict_res_list.append(predict_class)
            print(f"img:{image_name} category is {predict_class}")
    res_df = {"Id": image_name_list, "Category": predict_res_list}
    df = pd.DataFrame(res_df)
    df.to_csv("./res_resnet18_0.0001.csv", index=None)


def plot_loss():
    import matplotlib.pyplot as plt
    df_loss = pd.read_csv('./resnet18_0.0001_loss.csv')
    plt.plot(df_loss)
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.title('resnet18 loss curve')
    plt.show()


if __name__ == '__main__':
    if torch.cuda.is_available():
        print("current cuda device", torch.cuda.current_device())
        print(torch.cuda.get_device_name(device))

    # train the model
    train()

    # # test the model
    # test()

    # # plot loss curve
    # plot_loss()

