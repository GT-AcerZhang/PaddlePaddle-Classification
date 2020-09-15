import json
import os
import random


# 打乱数据
import shutil


def shuffle_data(data_list_path):
    with open(data_list_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        random.shuffle(lines)
        print("训练数据：%d 张" % len(lines))
    with open(data_list_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)


# 生成数据
def run(data_dir, train_list_path, test_list_path, label_path):
    f_train = open(train_list_path, 'w', encoding='utf-8')
    f_test = open(test_list_path, 'w', encoding='utf-8')
    f_label = open(label_path, 'w', encoding='utf-8')
    label_dict = dict()
    class_label = 0
    class_dirs = os.listdir(data_dir)
    for class_dir in class_dirs:
        if class_dir not in label_dict:
            label_dict[class_dir] = class_label
        class_sum = 0
        path = data_dir + "/" + class_dir
        img_paths = os.listdir(path)
        for img_path in img_paths:
            name_path = path + '/' + img_path
            if class_sum % 10 == 0:
                f_test.write(name_path + " %d" % class_label + "\n")
            else:
                f_train.write(name_path + " %d" % class_label + "\n")
            class_sum += 1
        class_label += 1
    label_str = str(label_dict).replace("'", '"')
    f_label.write(label_str)
    f_label.close()
    f_train.close()
    f_test.close()
    print('create data list done!')

    # 打乱数据
    shuffle_data(train_list_path)


def change_dir_name(data_dir, train_list_path, test_list_path, label_path):
    f_train = open(train_list_path, 'w', encoding='utf-8')
    f_test = open(test_list_path, 'w', encoding='utf-8')
    # with open(label_path, 'r', encoding='utf-8') as f_label:
    #     label = json.loads(f_label.read())
    # dirs = os.listdir(data_dir)
    # for dir in dirs:
    #     shutil.move(os.path.join(data_dir, dir), os.path.join(data_dir, str(label[dir])))

    class_dirs = os.listdir(data_dir)
    for class_dir in class_dirs:
        class_sum = 0
        path = data_dir + "/" + class_dir
        img_paths = os.listdir(path)
        for img_path in img_paths:
            name_path = path + '/' + img_path
            if class_sum % 10 == 0:
                f_test.write(name_path + " %s" % class_dir + "\n")
            else:
                f_train.write(name_path + " %s" % class_dir + "\n")
            class_sum += 1
    f_train.close()
    f_test.close()
    print('create data list done!')

    # 打乱数据
    shuffle_data(train_list_path)


if __name__ == '__main__':
    data_dir = 'dataset/images'
    train_list = 'dataset/train_list.txt'
    test_list = 'dataset/test_list.txt'
    label_file = 'dataset/labels.txt'
    # run(data_dir, train_list, test_list, label_file)
    change_dir_name(data_dir, train_list, test_list, label_file)
