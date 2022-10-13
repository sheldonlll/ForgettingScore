import datetime
import time
import torch
from torchvision import datasets
from torchvision.transforms import transforms
from torch import optim
import os
from matplotlib import pyplot as plt
import numpy as np

from mymodule.model import resnet34

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

def load_data(test_batch_size = 32, train_batch_size = 32, download = False, shuffle = True, ret_custom_all_data = False, category_num = 0, data_transform = {
        "train": transforms.Compose([transforms.Resize((32, 32)),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),

        "val": transforms.Compose([transforms.Resize((32, 32)),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }):
    
    cifar10_train = datasets.CIFAR10(root = 'datasets', train = True, download = download, transform = data_transform["train"])
    cifar10_test = datasets.CIFAR10(root = 'datasets', train = False, download = download, transform = data_transform["val"])
    # cifar10_train size: 50000        cifar10_test_size: 10000 
    print(f"cifar10_train size: {len(cifar10_train)} \t cifar10_test_size: {len(cifar10_test)}")
    kwargs = {'num_workers': 6, 'pin_memory': True} if torch.cuda.is_available() else {}
    cifar10_train_dataloader = torch.utils.data.DataLoader(cifar10_train, batch_size = train_batch_size, shuffle = shuffle, num_workers = kwargs["num_workers"], pin_memory = kwargs["pin_memory"])
    cifar10_test_dataloader = torch.utils.data.DataLoader(cifar10_test, batch_size = test_batch_size, shuffle = shuffle, num_workers = kwargs["num_workers"], pin_memory = kwargs["pin_memory"])

    if ret_custom_all_data == False:
        x, label = iter(cifar10_train_dataloader).next()
        test_x, test_label = iter(cifar10_test_dataloader).next()
        return x, label, test_x, test_label, cifar10_train_dataloader, cifar10_test_dataloader

    else:
        train_data_iter = iter(cifar10_train_dataloader)
        train_image, train_label = train_data_iter.next()

        img_all_train = torch.zeros(500, 3, 32, 32) # 存放train_image中所有标签是参数category_num的图片作为训练数据集
        train_image_num = 0 # img_all_train数组的当前数量/下标，最多500张

        for i in range(train_batch_size):
            if (train_label[i] == category_num):
                img_all_train[train_image_num] = train_image[i]
                train_image_num += 1
            if train_image_num == 500:
                break
        


        test_data_iter = iter(cifar10_test_dataloader)
        test_image, test_label = test_data_iter.next()

        img_all_test = torch.zeros(100, 3, 32, 32) # 存放test_image中所有标签是参数category_num的图片作为训练数据集
        test_image_num = 0 # img_all_train数组的当前 数量/下标，最多100张

        for i in range(test_batch_size):
            if (test_label[i] == category_num):
                img_all_test[test_image_num] = test_image[i]
                test_image_num += 1
            if test_image_num == 100:
                break
        
        return img_all_train, img_all_test # shape: (500, 3, 224, 224), (100, 3, 224, 224)



def train_predict_save_per_epoch(cifar10_train, cifar10_test, epoches, checkpoint_path = "./resnet_cifar10_cpts/"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet34().to(device)

    criteon = torch.nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    loss_lst = []
    acc_lst = []
    acc_detail_dict = {}
    tot_epoch = 0
    for epoch in range(epoches):
        model.train()
        loss = torch.tensor(-1.0)
        lossMIN = 0x3fff
        launchTimestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
        for _batch_idx, (x, label) in enumerate(cifar10_train):
            x, label = x.to(device), label.to(device)
            try:
                logits = model(x)
                loss = criteon(logits, label)
                lossMIN = min(lossMIN, loss)
                optimizer.zero_grad()
                loss.backward() 
                optimizer.step()
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print("WARNING: out of memory")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise exception
            
        loss_lst.append(lossMIN.cpu().detach().numpy())
        print(f"epoch: {epoch + 1}, current epoch min loss: {lossMIN.item()}")

        model.eval()
        with torch.no_grad():
            tot_correct = 0
            tot_num = 0
            for x, label in cifar10_test:
                x, label = x.to(device), label.to(device)
                logits = model(x) # [b, 10]
                pred = logits.argmax(dim = 1)
                res = torch.eq(pred, label)
                if acc_detail_dict.get(tot_epoch) == None:
                    acc_detail_dict[tot_epoch] = res.int().tolist()
                else:
                    acc_detail_dict[tot_epoch].append(res.int().tolist())
                tot_correct += res.float().sum().item()
                tot_num += x.shape[0]
            acc = tot_correct / tot_num
            print(f"epoch: {epoch + 1}, accuracy: {acc}")
        acc_lst.append(acc)
        torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN,
                            'optimizer': optimizer.state_dict()},
                           checkpoint_path + '/m-' + launchTimestamp + '-' + str("%.4f" % lossMIN) + '.pth.tar')
        tot_epoch += 1
        
    x_epoches = np.arange(tot_epoch)
    loss_lst = np.array(loss_lst)
    acc_lst = np.array(acc_lst)
    plt.plot(x_epoches, loss_lst, label = "loss line")
    plt.plot(x_epoches, acc_lst, label = "accuracy line")
    plt.xlabel("epoch")
    plt.legend()
    plt.show()
    with open("acc_per_epoch_detail_lst.txt", "w+") as f:
        cur_data = ""
        for key in sorted(acc_detail_dict.keys()):
            cur_data += "epoch_" + str(key) + " = " + str(acc_detail_dict[key]) + "\n"
        f.write(cur_data)
    return model


def main():
    #   cifar10_train size: 50000        cifar10_test_size: 10000 
    #   torch.Size([32, 3, 32, 32])   torch.Size([32])   torch.Size([32, 3, 32, 32])   torch.Size([32])
    torch.cuda.empty_cache()
    data_transform = {
            "train": transforms.Compose([transforms.Resize(256),
                            transforms.CenterCrop(224), #TODO
                            transforms.ToTensor(),# converts images loaded by Pillow into PyTorch tensors.
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),

            "val": transforms.Compose([transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        } # transform incoming images into a Pytorch Tensor

    x, label, test_x, test_label, cifar10_train_dataloader, cifar10_test_dataloader = load_data(test_batch_size = 256, train_batch_size = 1280, download = True, category_num = 0, ret_custom_all_data=False, data_transform=data_transform)

    print(f"train shape: {x.shape} train label shape: {label.shape} test shape: {test_x.shape} test label shape: {test_label.shape}")
    train_predict_save_per_epoch(cifar10_train_dataloader, cifar10_test_dataloader, epoches = 50, checkpoint_path="./resnet_cifar10_cpts/")

    '''
    img_all_train, img_all_test = load_data(test_batch_size = 1000, train_batch_size = 5000, category_num = 0)
    train_predict(img_all_train, img_all_test, 1000)
    '''


if __name__ == "__main__":
    main()


'''
Files already downloaded and verified
Files already downloaded and verified
cifar10_train size: 50000 	 cifar10_test_size: 10000
train shape: torch.Size([1280, 3, 224, 224]) train label shape: torch.Size([1280]) test shape: torch.Size([256, 3, 224, 224]) test label shape: torch.Size([256])
epoch: 1, current epoch min loss: 1.4466698169708252
epoch: 1, accuracy: 0.3034
epoch: 2, current epoch min loss: 1.0604150295257568
epoch: 2, accuracy: 0.3809
epoch: 3, current epoch min loss: 0.9211944341659546
epoch: 3, accuracy: 0.5064
epoch: 4, current epoch min loss: 0.7223517298698425
epoch: 4, accuracy: 0.6084
epoch: 5, current epoch min loss: 0.5098822712898254
epoch: 5, accuracy: 0.6444
epoch: 6, current epoch min loss: 0.5415691137313843
epoch: 6, accuracy: 0.6691
epoch: 7, current epoch min loss: 0.46661216020584106
epoch: 7, accuracy: 0.6889
epoch: 8, current epoch min loss: 0.3920561671257019
epoch: 8, accuracy: 0.7494
epoch: 9, current epoch min loss: 0.34063780307769775
epoch: 9, accuracy: 0.7427
epoch: 10, current epoch min loss: 0.30405113101005554
epoch: 10, accuracy: 0.7569
epoch: 11, current epoch min loss: 0.2511935830116272
epoch: 11, accuracy: 0.7499
epoch: 12, current epoch min loss: 0.20740923285484314
epoch: 12, accuracy: 0.7986
epoch: 13, current epoch min loss: 0.17469438910484314
epoch: 13, accuracy: 0.7972
epoch: 14, current epoch min loss: 0.14681853353977203
epoch: 14, accuracy: 0.7261
epoch: 15, current epoch min loss: 0.1116250529885292
epoch: 15, accuracy: 0.7686
epoch: 16, current epoch min loss: 0.09368099272251129
epoch: 16, accuracy: 0.7912
epoch: 17, current epoch min loss: 0.07605766505002975
epoch: 17, accuracy: 0.7387
epoch: 18, current epoch min loss: 0.05136311799287796
epoch: 18, accuracy: 0.7838
epoch: 19, current epoch min loss: 0.03625984489917755
epoch: 19, accuracy: 0.8373
epoch: 20, current epoch min loss: 0.01806572824716568
epoch: 20, accuracy: 0.8314
epoch: 21, current epoch min loss: 0.021827273070812225
epoch: 21, accuracy: 0.8443
epoch: 22, current epoch min loss: 0.030667927116155624
epoch: 22, accuracy: 0.7602
epoch: 23, current epoch min loss: 0.03276913985610008
epoch: 23, accuracy: 0.8148
epoch: 24, current epoch min loss: 0.023377224802970886
epoch: 24, accuracy: 0.8256
epoch: 25, current epoch min loss: 0.011851618997752666
epoch: 25, accuracy: 0.8364
epoch: 26, current epoch min loss: 0.0022362719755619764
epoch: 26, accuracy: 0.8502
epoch: 27, current epoch min loss: 0.0011447076685726643
epoch: 27, accuracy: 0.866
epoch: 28, current epoch min loss: 0.00051883578998968
epoch: 28, accuracy: 0.867
epoch: 29, current epoch min loss: 0.00031393219251185656
epoch: 29, accuracy: 0.8708
epoch: 30, current epoch min loss: 0.00039345351979136467
epoch: 30, accuracy: 0.8663
epoch: 31, current epoch min loss: 0.0005601632874459028
epoch: 31, accuracy: 0.8664
epoch: 32, current epoch min loss: 0.0002762923249974847
epoch: 32, accuracy: 0.8656
epoch: 33, current epoch min loss: 0.00020346662495285273
epoch: 33, accuracy: 0.8703
epoch: 34, current epoch min loss: 0.0001524393301224336
epoch: 34, accuracy: 0.8672
epoch: 35, current epoch min loss: 0.0002080528938677162
epoch: 35, accuracy: 0.8645
epoch: 36, current epoch min loss: 0.00018332366016693413
epoch: 36, accuracy: 0.8714
epoch: 37, current epoch min loss: 0.00022438474115915596
epoch: 37, accuracy: 0.8456
epoch: 38, current epoch min loss: 0.0005740060005337
epoch: 38, accuracy: 0.8636
epoch: 39, current epoch min loss: 0.0006648817216046154
epoch: 39, accuracy: 0.7595
epoch: 40, current epoch min loss: 0.05758208781480789
epoch: 40, accuracy: 0.7148
epoch: 41, current epoch min loss: 0.05498562008142471
epoch: 41, accuracy: 0.7538
epoch: 42, current epoch min loss: 0.019564732909202576
epoch: 42, accuracy: 0.8154
epoch: 43, current epoch min loss: 0.02125411294400692
epoch: 43, accuracy: 0.8122
epoch: 44, current epoch min loss: 0.010802713222801685
epoch: 44, accuracy: 0.833
epoch: 45, current epoch min loss: 0.008106651715934277
epoch: 45, accuracy: 0.8372
epoch: 46, current epoch min loss: 0.012133958749473095
epoch: 46, accuracy: 0.8139
epoch: 47, current epoch min loss: 0.013578077778220177
epoch: 47, accuracy: 0.8315
epoch: 48, current epoch min loss: 0.0061781094409525394
epoch: 48, accuracy: 0.839
epoch: 49, current epoch min loss: 0.0023734397254884243
epoch: 49, accuracy: 0.8554
epoch: 50, current epoch min loss: 0.002421722747385502
epoch: 50, accuracy: 0.8349

see accuracy, loss - per epoch line in accuracy, loss - epoch line.png

ResNet(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace=True)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (layer1): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU()
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (1): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU()
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (2): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU()
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer2): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU()
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU()
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (2): BasicBlock(
      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU()
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (3): BasicBlock(
      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU()
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer3): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU()
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU()
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (2): BasicBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU()
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (3): BasicBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU()
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (4): BasicBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU()
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (5): BasicBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU()
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer4): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU()
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU()
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (2): BasicBlock(
      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU()
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (fc): Linear(in_features=512, out_features=1000, bias=True)
)
'''