from PIL import Image
import numpy as np
import logging
import os
import time
import sys

#打印日志到控制台和log_path下的txt文件
def get_logger( log_path='log_path'):
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    timer=time.strftime("%Y-%m-%d-%H-%M-%S_", time.localtime())
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(levelname)s]   %(asctime)s    %(message)s')
    txthandle = logging.FileHandler((log_path+'/'+timer+'log.txt'))
    txthandle.setFormatter(formatter)
    logger.addHandler(txthandle)
    return logger

#将输入路径的上两级路径加入系统
def set_projectpath(current_path):
    curPath = os.path.abspath(current_path)
    #curPath = os.path.abspath(os.path.dirname(__file__))
    rootPath = os.path.split(curPath)[0]
    sys.path.append(rootPath)
    rootPath = os.path.split(rootPath)[0]
    sys.path.append(rootPath)


def concatImage(images,mode="L"):
	if not isinstance(images, list):
		raise Exception('images must be a  list  ')
	count=len(images)
	size= Image.fromarray(images[0]).size
	target = Image.new(mode, (size[0] * count, size[1] * 1))
	for i  in  range(count):
		image = Image.fromarray(images[i]).resize(size, Image.BILINEAR)
		target.paste(image, (i*size[0], 0, (i+1)*size[0], size[1]))
	return target


def listData2(data_dir,test_ratio=0.4):
    example_dirs = [x[1] for x in os.walk(data_dir)][0]
    print({example_dirs[i]: i for i in range(len(example_dirs))})

    example_lists = {x: os.listdir('{}/{}/Imgs'.format(data_dir, x)) for x in example_dirs}

    Positive_examples_train = []
    Negative_examples_train = []
    Positive_examples_valid = []
    Negative_examples_valid = []
    for i in range(len(example_dirs)):
        example_dir = example_dirs[i]
        example_list = example_lists[example_dir]
        # 过滤label图片
        example_list = [item for item in example_list if item.endswith('jpg')]
        # 训练数据
        train_test_offset=int(np.floor(len(example_list)*(1-test_ratio)))
        examples = [[example_dir + '/Imgs/' + x, example_dir + '/Imgs/' + x.split('.')[0] + '.png'] for x in example_list]
        
        if example_dir == 'MT_Free':
             Positive_examples_train.extend(examples[:train_test_offset])
             Positive_examples_valid.extend(examples[train_test_offset:])
        else:
             Negative_examples_train.extend(examples[:train_test_offset])
             Negative_examples_valid.extend(examples[train_test_offset:])

    print(len(Positive_examples_train),len(Negative_examples_train))
    print(Positive_examples_train[0],Negative_examples_train[0])


if __name__ == '__main__':
    listData2("../Surface-Defect-Detection/Magnetic-Tile-Defect")