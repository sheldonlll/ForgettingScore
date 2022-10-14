import json
from itertools import chain
from tool.forgettingscore import calculate_two_epoch_forgettingscore
import os

all_epoch_detail_dict = {}


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

    for i in range(len(data)):
        # 1.
        epoch_idx, current_epoch_accuracy_detail_str = data[i].split(" = ")[0], data[i].split(" = ")[-1]
        idx = current_epoch_accuracy_detail_str.find("[", 1)
        current_epoch_accuracy_detail_seperate_by_epoch_lst = json.loads("[" + current_epoch_accuracy_detail_str[:idx - 2] + "], " + current_epoch_accuracy_detail_str[idx:])
        
        # 2.
        current_epoch_accuracy_detail_lst = list(chain(*current_epoch_accuracy_detail_seperate_by_epoch_lst))
        # print(len(current_epoch_accuracy_detail_lst)) #10000
        all_epoch_detail_dict[epoch_idx] = current_epoch_accuracy_detail_lst
    
    # print(all_epoch_detail_dict)



def Algorithm_1_Computing_forgetting_statistics(to_shuffle = False):
    tot_forgetting_score = []
    global all_epoch_detail_dict 
    all_epoch_detail_dict = sorted(all_epoch_detail_dict.keys(), key = lambda x: int(x.split("_")[-1])) 
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
        per_epoch_data_len = len(all_epoch_detail_dict.get(all_epoch_detail_dict[current_batch_idx]))
        for i in range(per_epoch_data_len):
            current_acc = all_epoch_detail_dict.get(all_epoch_detail_dict[current_batch_idx])[i]
            next_acc = all_epoch_detail_dict.get(all_epoch_detail_dict[next_batch_idx])[i]
            pass



def main():
    file_data = None
    file_path = input("输入每经过一个epoch后, 记录有每个测试数据的标签正误的文件(文件格式：epoch_0 = [...]\nepoch_1 = [...])路径：")
    if os.path.exists(file_path) == False:
        file_path = "C:\\Users\\Sherlock\\Desktop\\pycodes\\ForgettingScore\\acc_per_epoch_detail_lst.txt"

        with open(file_path, "r") as f:
            file_data = f.readlines()

        prepare_data(file_data) #处理格式错误
        Algorithm_1_Computing_forgetting_statistics(to_shuffle = False) #计算forgettingscore

    else:
        with open(file_path, "r") as f:
            file_data = f.readlines()
        Algorithm_1_Computing_forgetting_statistics(to_shuffle = False) #计算forgettingscore


if __name__ == "__main__":
    main()


