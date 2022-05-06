from nltk.stem.porter import *
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.svm import SVC
POS_DICT = []
BIO_DICT = []
PATH_DICT = []

def index_list(dict, key):
    if key in dict:
        return dict.index(key)
    else:
        dict.append(key)
        return dict.index(key)

def read_data(pth):
    pairs_data = []
    sentence = []
    with open(pth, 'r') as file:
        lines_in = file.read().split('\n')
        for line in lines_in:
            if line == '':
                if sentence == '' or len(sentence) < 3:
                    sentence = []
                    continue
                # print("s", sentence)
                pairs_data += find_pair(sentence)
                sentence = []
                continue
            else:
                sentence.append(line)
    # print("pairs_data", pairs_data)
    return pairs_data

def find_pair(sentence):
    pairs_data = []
    whole_POS_path = []
    whole_BIO_path = [] # ------TODO: introduce BIO path-------
    for i, word in enumerate(sentence):
        data = word.split('\t')
        whole_POS_path.append(data[1])
        whole_BIO_path.append(data[2])
        if len(data) < 6:
            continue
        if data[5] == "ARG1":
            ARG_POS = data[1]
            ARG_BIO = data[2]
            ARG_num = data[3]

        elif data[5] == "PRED":
            PRED_POS = data[1]
            PRED_BIO = data[2]
            PRED_num = data[3]
            PRED_index = i

    for i, word in enumerate(sentence):
        data = word.split('\t')
        if i == PRED_index:
            continue
        if i < PRED_index:
            current_path = [whole_POS_path[x] for x in range(i, PRED_index+1)]
        else:
            current_path = [whole_POS_path[x] for x in range(PRED_index, i+1)]
        current_path = ','.join(current_path)
        if len(data) == 6 and data[5] == "ARG1":
            role = 1
        else:
            role = 0
        pairs_data.append([index_list(POS_DICT, PRED_POS)] + [index_list(BIO_DICT, PRED_BIO)]
                          + [index_list(POS_DICT, data[1])] + [index_list(BIO_DICT, data[2])]
                          + [int(PRED_num) - int(data[3])] + [index_list(PATH_DICT, current_path)]
                          + [role])
    return pairs_data

def onehot_pair(pairs, withpth=1):
    res = []
    for pair in pairs:
        t1 = [0 for x in range(len(POS_DICT))]
        t1[pair[0]] = 1
        t2 = [0 for x in range(len(BIO_DICT))]
        t2[pair[1]] = 1
        t3 = [0 for x in range(len(POS_DICT))]
        t3[pair[2]] = 1
        t4 = [0 for x in range(len(BIO_DICT))]
        t4[pair[3]] = 1
        if withpth:
            t5 = [0 for x in range(len(PATH_DICT))]
            t5[pair[4]] = 1
            res.append(t1 + t2 + t3 + t4 + t5 + [pair[6]] )
        else:
            res.append(t1 + t2 + t3 + t4 + [pair[6]])
    return np.array(res)

if __name__ == "__main__":

    pair_train = np.load("train.npy")
    x = pair_train[:, 0:-1]
    y = pair_train[:, -1]

    print(x.shape)
    print(y.shape)

    # pair_test = onehot_pair(np.load("dev.npy"), withpth=0)
    pair_test = np.load("test.npy")
    x_test = pair_test[:, 0:-1]
    y_test = pair_test[:, -1]

    print("training DT")
    clf = RandomForestClassifier(n_jobs=4)  # n_jobs=2是线程数
    clf.fit(x, y)  # 训练过程
    y_pred_DT = clf.predict(x_test) # 获取测试数据预测结果
    print(confusion_matrix(y_test, y_pred_DT))
    print(accuracy_score(y_test,y_pred_DT))

    N = 0



    with open("%-test", 'r') as file_in:
        with open("partitive.txt", 'w') as file_out:
            lines_in = file_in.read().split('\n')
            for line in lines_in:
                data = line.split("\t")
                if line == '':
                    file_out.write("\n")
                    continue
                if len(data) == 6 and data[5] == "PRED":
                    file_out.write(line + "\n")
                    continue
                if len(data) < 6:
                    data.append("")
                if y_pred_DT[N] == 1:
                    data[5] = "ARG1"
                else:
                    data[5] = ""
                N += 1
                file_out.write("\t".join(data) + "\n")
    assert N == len(y_pred_DT)


