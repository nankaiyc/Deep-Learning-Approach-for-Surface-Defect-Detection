import  tensorflow as tf
import numpy as np
import shutil
import  os
from data_manager import DataManager
from  model import  Model
from config import  *
import utils
from datetime import datetime
from tqdm import tqdm

class Agent(object):
	def __init__(self,param):

		self.__sess=tf.Session()
		self.__Param=param
		self.init_datasets()  #初始化数据管理器
		self.model=Model(self.__sess,self.__Param) #建立模型
		self.logger=utils.get_logger(param["Log_dir"])
	def run(self):
		if self.__Param["mode"] is "training":
			train_mode= self.__Param["train_mode"]
			self.train(train_mode)
		elif self.__Param["mode"] is "testing":
			self.test()
		elif self.__Param["mode"] is "savePb":
			raise Exception(" this  mode is incomplete ")
		else:
			print("got a unexpected mode ,please set the mode  'training', 'testing' or 'savePb' ")

	def init_datasets(self):
		# self.Positive_data_list,self.Negative_data_list=self.listData1(self.__Param["data_dir"])
		self.Positive_data_list,self.Negative_data_list=self.listDataMagneticTile(self.__Param["data_dir"])
		if self.__Param["mode"] is "training":
			self.DataManager_train_Positive = DataManager(self.Positive_data_list, self.__Param)
			self.DataManager_train_Negative = DataManager(self.Negative_data_list, self.__Param)
		elif self.__Param["mode"] is "testing":
			self.DataManager_test_Positive = DataManager(self.Positive_data_list, self.__Param,shuffle=False)
			self.DataManager_test_Negative = DataManager(self.Negative_data_list, self.__Param,shuffle=False)
		elif self.__Param["mode"] is "savePb":
			pass
		else:
			raise Exception('got a unexpected  mode ')

	def train(self,mode):
		if mode not in ["segment","decision","total","data"]:
			raise Exception('got a unexpected  training mode ,options :{segment,decision}')
		with self.__sess.as_default():
			self.logger.info('start training {} net with step {}'.format(mode, self.model.step))
			for i in tqdm(range(self.model.step, self.__Param["epochs_num"] + self.model.step)):
				#epoch start
				iter_loss = 0
				for index  in  range(2):
					#batch start
					if index==0 :
						data_manager = self.DataManager_train_Positive
					else:
						data_manager = self.DataManager_train_Negative

					for batch in range(data_manager.number_batch):
						#corss training the positive sample and negative sample
						img_batch, label_pixel_batch, label_batch, file_name_batch, = self.__sess.run(data_manager.next_batch)
						loss_value_batch=0

						if mode == "segment":
							_, loss_value_batch = self.__sess.run([self.model.optimize_segment,self.model.loss_pixel],
								feed_dict={self.model.Image: img_batch,
										   self.model.PixelLabel: label_pixel_batch})
						elif mode =="decision":
							_, loss_value_batch = self.__sess.run([self.model.optimize_decision, self.model.loss_class],
												feed_dict={self.model.Image: img_batch,
														   self.model.Label: label_batch})
						elif mode == "total":
							_, loss_value_batch = self.__sess.run([self.model.optimize_total, self.model.loss_total],
														feed_dict={self.model.Image: img_batch,
																   self.model.PixelLabel: label_pixel_batch,
																   self.model.Label: label_batch})
						else:
							print(img_batch, label_pixel_batch,label_batch, file_name_batch)
							return
						
						iter_loss+= loss_value_batch
						#可视化
						if i % self.__Param["valid_frequency"] == 0 and i>0:
							mask_batch = self.__sess.run(self.model.mask, feed_dict={self.model.Image: img_batch})
							save_dir = "./visualization/training_epoch-{}".format(i)
							self.visualization(img_batch, label_pixel_batch, mask_batch, file_name_batch,save_dir)

				self.logger.info('epoch:[{}] ,train_mode:{}, loss: {}'.format(self.model.step, mode,iter_loss))
				#保存模型
				if i % self.__Param["save_frequency"] == 0 or i==self.__Param["epochs_num"] + self.model.step-1:
					self.model.save()
				# #验证
				# if i % self.__Param["valid_frequency"] == 0 and i>0:
				# 	self.valid()
				self.model.step += 1


	def test(self):
		#anew a floder to save visualization
		visualization_dir="./visualization/test"
		if not os.path.exists(visualization_dir):
			os.makedirs(visualization_dir)
		with self.__sess.as_default():
			self.logger.info('start testing')
			count=0
			count_TP = 0  # 真正例
			count_FP = 0  # 假正例
			count_TN = 0  # 真反例
			count_FN = 0  # 假反例
			DataManager=[self.DataManager_test_Positive,self.DataManager_test_Negative]
			for index in range(2):
				for batch in tqdm(range(DataManager[index].number_batch)):
					img_batch, label_pixel_batch,label_batch, file_name_batch,  = self.__sess.run(DataManager[index].next_batch)
					mask_batch ,output_batch= self.__sess.run([self.model.mask,self.model.output_class],
						feed_dict={self.model.Image: img_batch,})
					self.visualization(img_batch, label_pixel_batch,mask_batch, file_name_batch,save_dir=visualization_dir)
					for i, filename in enumerate(file_name_batch):
						count +=1
						if label_batch[i] == 0 and output_batch[i] == 0:
							count_TP += 1
						elif label_batch[i] == 0:
							count_FN += 1
							self.logger.info("Image {} is label {} but output {}".format(file_name_batch[i], label_batch[i], output_batch[i]))
						elif output_batch[i] == 0:
							count_FP += 1
							self.logger.info("Image {} is label {} but output {}".format(file_name_batch[i], label_batch[i], output_batch[i]))
						else:
							count_TN += 1
			# 准确率
			accuracy = (count_TP + count_TN) / count
			# 查准率
			prescision = count_TP / (count_TP + count_FP)
			# 查全率
			recall = count_TP / (count_TP + count_FN)
			self.logger.info(" total number of samples = {}".format(count))
			self.logger.info("positive = {}".format(count_TP + count_FN))
			self.logger.info("negative = {}".format(count_FP + count_TN))
			self.logger.info("TP = {}".format(count_TP ))
			self.logger.info("NP = {}".format(count_FP))
			self.logger.info("TN = {}".format(count_TN ))
			self.logger.info("FN = {}".format(count_FN ))
			self.logger.info("accuracy(准确率) = {:.4f}".format((count_TP + count_TN) / count))
			self.logger.info("prescision（查准率） = {:.4f}".format(prescision))
			self.logger.info("recall（查全率） = {:.4f}".format(recall))
			self.logger.info("the visualization saved in {}".format(visualization_dir))
	def valid(self):
		pass

	def visualization(self,img_batch,label_pixel_batch,mask_batch,filenames,save_dir="./visualization"):
		#anew a floder to save visualization
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)
		for i, filename in enumerate(filenames):
			filename = str(filename).split("'")[-2].replace("/","_")
			mask=np.array(mask_batch[i]).squeeze(2)*255
			image=np.array(img_batch[i]).squeeze(2)
			label_pixel = np.array(label_pixel_batch[i]).squeeze(2)*255
			img_visual=utils.concatImage([image,label_pixel,mask])
			visualization_path = os.path.join(save_dir,filename)
			img_visual.save(visualization_path)


	def listData(self,data_dir):
		"""# list the files  of  the currtent  floder of  'data_dir' 	,subfoders are not included.
		:param data_dir:
		:return:  list of files
		"""
		data_list=os.listdir(data_dir)
		data_list=[x[2] for x in os.walk(data_dir)][0]
		data_size=len(data_list)
		return data_list,data_size

	def listData1(self,data_dir,test_ratio=0.4,positive_index=POSITIVE_KolektorSDD):
		""" this function is designed for the Dataset of KolektorSDD,
			the positive samples and negative samples will be divided into two lists
		:param  data_dir:  the  data folder   of KolektorSDD
		:param  test_ratio: the proportion of test set
		:param positive_index:   the  list  of  index of   every subfolders' positive samples
		:return:    the list of  the positive samples and the list of negative samples
		"""
		example_dirs = [x[1] for x in os.walk(data_dir)][0]
		example_lists = {os.path.basename(x[0]): x[2] for x in os.walk(data_dir)}
		train_test_offset=np.floor(len(example_lists)*(1-test_ratio))
		Positive_examples_train = []
		Negative_examples_train = []
		Positive_examples_valid = []
		Negative_examples_valid = []
		for i in range(len(example_dirs)):
			example_dir = example_dirs[i]
			example_list = example_lists[example_dir]
			# 过滤label图片
			example_list = [item for item in example_list if "label" not in item]
			# 训练数据
			if i < train_test_offset:
				for j in range(len(example_list)):
					example_image = example_dir + '/' + example_list[j]
					example_label = example_image.split(".")[0] + "_label.bmp"
					# 判断是否是正样本
					index = example_list[j].split(".")[0][-1]
					if index in positive_index[i]:
						Positive_examples_train.append([example_image, example_label])
					else:
						Negative_examples_train.append([example_image, example_label])
			else:
				for j in range(len(example_list)):
					example_image = example_dir + '/' + example_list[j]
					example_label = example_image.split(".")[0] + "_label.bmp"
					index=example_list[j].split(".")[0][-1]
					if index in positive_index[i]:
						Positive_examples_valid.append([example_image, example_label])
					else:
						Negative_examples_valid.append([example_image, example_label])
		if self.__Param["mode"] is "training":
			return Positive_examples_train,Negative_examples_train
		if self.__Param["mode"] is "testing":
			return Positive_examples_valid,Negative_examples_valid


	def listDataMagneticTile(self,data_dir,test_ratio=0.2):
		example_dirs = [x[1] for x in os.walk(data_dir)][0]

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
			data_len = len(example_list)
			train_test_offset=int(np.floor(data_len*(1-test_ratio)))
			examples = [[example_dir + '/Imgs/' + x, example_dir + '/Imgs/' + x.split('.')[0] + '.png'] for x in example_list]
			
			# slice 1/10, remeber to del
			# alpha = 0.03
			# examples = examples[:int(data_len*alpha)]
			# train_test_offset = int(train_test_offset*alpha)
			
			if example_dir == 'MT_Free':
				Positive_examples_train.extend(examples[:train_test_offset])
				Positive_examples_valid.extend(examples[train_test_offset:])
			else:
				Negative_examples_train.extend(examples[:train_test_offset])
				Negative_examples_valid.extend(examples[train_test_offset:])
	     
		if self.__Param["mode"] is "training":
			return Positive_examples_train,Negative_examples_train
		if self.__Param["mode"] is "testing":
			return Positive_examples_valid,Negative_examples_valid





