import os
import random

path_to_features = 'data/Features'
path_to_valid = 'data/validation'
categories = os.listdir(path_to_features)

for category in categories:
    print('------------------- category : ', category, '-----------------')
    path_to_category = os.path.join(path_to_features, category)
    path_to_category_valid = os.path.join(path_to_valid, category)
    num_files = len(os.listdir(path_to_category))
    num_files_to_valid = int(num_files/10)
    file_list = os.listdir(path_to_category)
    if not os.path.isdir(path_to_category_valid):
        os.mkdir(path_to_category_valid)
    for i in range(num_files_to_valid):
        try:
            file_to_move = random.randint(0,num_files)
            file_to_move = file_list[file_to_move]
            file_to_move_path = os.path.join(path_to_category, file_to_move)
            print('moving file : ',file_to_move)
            path_to_move_to = os.path.join(path_to_category_valid, file_to_move)
            os.rename(file_to_move_path, path_to_move_to)
            print('file moved to ', path_to_move_to)
            num_files = len(os.listdir(path_to_category))
            file_list = os.listdir(path_to_category)
        except:
            print('error')
    print('************** category :', category, ' done ***************')
