import json
from itertools import chain
from tool.forgettingscore import calculate_two_epoch_forgettingscore
import os
from torchvision.transforms import transforms
from torchvision import datasets
import torch

from main_resnet34_cifar10 import train_predict_save_per_epoch


all_epoch_detail_dict = {}
len_per_epoch = 0


def prepare_data(data):
    '''
1. 处理文件数据中每个epoch第一个列表缺少的右中括号(main_resnet34_cifar10.py中列表的格式没写好, 在这里补充)
让它成为完整的二维列表(batch1 = (1, 0, 1, ...), batch2, ...)
2. 再展开成一位列表(1, 0, 1, ...)

idx: 769, 缺失的右括号的位置
40,每个epoch共有40个batch
256(每个batch的大小) 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 16       

idx: 769
40
256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 16       

idx: 769
40
256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 16       
    '''
    global len_per_epoch
    
    for i in range(len(data)):
        # 1.
        epoch_idx, current_epoch_accuracy_detail_str = data[i].split(" = ")[0], data[i].split(" = ")[-1]
        idx = current_epoch_accuracy_detail_str.find("[", 1)
        current_epoch_accuracy_detail_seperate_by_epoch_lst = json.loads("[" + current_epoch_accuracy_detail_str[:idx - 2] + "], " + current_epoch_accuracy_detail_str[idx:])
        
        # 2.
        current_epoch_accuracy_detail_lst = list(chain(*current_epoch_accuracy_detail_seperate_by_epoch_lst))
        # print(len(current_epoch_accuracy_detail_lst)) #10000
        len_per_epoch = len(current_epoch_accuracy_detail_lst)
        all_epoch_detail_dict[epoch_idx] = current_epoch_accuracy_detail_lst
    
    # print(all_epoch_detail_dict)


def Algorithm_1_Computing_forgetting_statistics(to_shuffle = False):
    """
    Acci(t) > Acci(t + 1): forgetting event happened, then scorei -= 2
    Acci(t) == Acci(t + 1) && Acci(t) == 0: then scorei -= 1
    Acci(t) == Acci(t + 1) && Acci(t) == 1: then scorei += 1
    Acci(t) < Acci(t + 1): learning event happened, then scorei += 2
    example i: the forgetting event never happened && learning event once happend => unforgettable example i
    example i: the forgetting event has happend once. => forgettable example
    the first learning event
    the first forgetting event
    """
    forgetting_event_happend_state = [False for _ in range(len_per_epoch)]
    learning_event_happend_state = [False for _ in range(len_per_epoch)]
    first_learning_event_happened_state = [-1 for _ in range(len_per_epoch)]
    first_forgetting_event_happend_state = [-1 for _ in range(len_per_epoch)]

    tot_forgetting_score = [0 for _ in range(len_per_epoch)]
    unforgettable_examples = []
    forgettable_examples = []
    global all_epoch_detail_dict 
    all_epoch_detail_dict = {k: v for k, v in sorted(all_epoch_detail_dict.items(), key=lambda item: item[0].split("_")[-1])}
    for current_batch_idx in range(len(all_epoch_detail_dict)): # while not training done, across all epoches
        next_batch_idx = min(current_batch_idx + 1, len(all_epoch_detail_dict) - 1)
        # print(all_epoch_detail_dict[current_batch_idx], all_epoch_detail_dict[next_batch_idx])
        '''
        epoch_0 epoch_1
        epoch_1 epoch_2
        epoch_2 epoch_3
            .....
        '''
        # no shuffle
        for i in range(len_per_epoch):
            current_acc = all_epoch_detail_dict["epoch_" + str(current_batch_idx)][i]
            next_acc = all_epoch_detail_dict["epoch_" + str(next_batch_idx)][i]
            
            if first_learning_event_happened_state[i] == -1 and current_acc < next_acc:
                first_learning_event_happened_state[i] = current_batch_idx
            if first_forgetting_event_happend_state[i] == -1 and learning_event_happend_state[i] and current_acc < next_acc:
                first_forgetting_event_happend_state[i] = current_batch_idx
            
            if current_acc > next_acc:
                tot_forgetting_score[i] -= 2
            elif current_acc == next_acc and current_acc == 0:
                tot_forgetting_score[i] -= 1
            elif current_acc == next_acc and current_acc == 1:
                tot_forgetting_score[i] += 1
            elif current_acc < next_acc:
                tot_forgetting_score[i] += 2
        
            forgetting_event_happend_state[i] = True if current_acc > next_acc else forgetting_event_happend_state[i]
            learning_event_happend_state[i] = True if current_acc < next_acc else learning_event_happend_state[i]
            

    for i in range(len_per_epoch):
        if forgetting_event_happend_state[i] == False and learning_event_happend_state[i] == True:
            unforgettable_examples.append(i)
        if forgetting_event_happend_state[i]:
            forgettable_examples.append(i)
    
    return tot_forgetting_score, unforgettable_examples, forgettable_examples, first_learning_event_happened_state, first_forgetting_event_happend_state


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
    
    x, label = iter(cifar10_train_dataloader).next()
    return x, label, cifar10_train_dataloader, cifar10_test_dataloader


def handl_train_data(tot_forgetting_score, unforgettable_examples, forgettable_examples, first_learning_event_happened_state, first_forgetting_event_happend_state):
    x, label, cifar10_train_dataloader = load_data(test_batch_size = 256, train_batch_size = 1280, shuffle = False, download = False, category_num = 0, ret_custom_all_data=False)
    # todo
    # select cifar10_train_data by tot_forgetting_score, unforgettable_examples, forgettable_examples, first_learning_event_happened_state, first_forgetting_event_happend_state
    # ...
    # finished
    return x, label, cifar10_train_dataloader


def train_again(x, label, cifar10_train_dataloader, cifar10_test_dataloader):
    train_predict_save_per_epoch(cifar10_train_dataloader, cifar10_test_dataloader, epoches = 50, checkpoint_path="./resnet_cifar10_cpts_version2/")


def main():
    file_data = None
    file_path = input("输入每经过一个epoch后, 记录有每个测试数据的标签正误的文件(文件格式：epoch_0 = [...] epoch_1 = [...])路径：")
    if os.path.exists(file_path) == False:
        file_path = "C:\\Users\\Sherlock\\Desktop\\pycodes\\ForgettingScore\\acc_per_epoch_detail_lst.txt"
        with open(file_path, "r") as f:
            file_data = f.readlines()

        prepare_data(file_data) #处理格式错误
        tot_forgetting_score, unforgettable_examples, forgettable_examples, first_learning_event_happened_state, first_forgetting_event_happend_state = Algorithm_1_Computing_forgetting_statistics(to_shuffle = False) #计算forgettingscore
        with open("forgetting_score_results.txt", "w+") as f:
            data = "tot_forgetting_score:\n" + str(tot_forgetting_score) + "\n" + "\nunforgettable_examples:\n"+ str(unforgettable_examples) + "\n" + "\nforgettable_examples:\n" + str(forgettable_examples) + "\nfirst_learning_event_happened_state:\n" + str(first_learning_event_happened_state) + "\nfirst_forgetting_event_happend_state:\n" + str(first_forgetting_event_happend_state)
            f.write(data)
        # x, label, cifar10_train_dataloader, cifar10_test_dataloader = handl_train_data(tot_forgetting_score, unforgettable_examples, forgettable_examples, first_learning_event_happened_state, first_forgetting_event_happend_state)
        # train_again(x, label, cifar10_train_dataloader, cifar10_test_dataloader)

    else:
        with open(file_path, "r") as f:
            file_data = f.readlines()
        Algorithm_1_Computing_forgetting_statistics(to_shuffle = False) #计算forgettingscore
        # x, label, cifar10_train_dataloader, cifar10_test_dataloader = handl_train_data(tot_forgetting_score, unforgettable_examples, forgettable_examples, first_forgetting_event_happend_state)
        # train_again(x, label, cifar10_train_dataloader, cifar10_test_dataloader)


if __name__ == "__main__":
    main()


