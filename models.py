# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import tensorflow as tf
import numpy as np
import cv2
from ops import *
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from utils import *

class AOD(object):
	"""Single Image Dehazing via Multi-Scale Convolutional Neural Networks"""
	def __init__(self, model_path):
		self.model_path = model_path
		self.graph_path = model_path+"/tf_graph/"
		self.save_path = model_path + "/saved_model/"
		self.output_path = model_path + "/results/"
		if not os.path.exists(model_path):
			os.makedirs(self.graph_path+"train/")
			os.makedirs(self.graph_path+"val/")
		
	def _debug_info(self):
		variables_names = [[v.name, v.get_shape().as_list()] for v in tf.trainable_variables()]
		print "Trainable Variables:"
		tot_params = 0
		for i in variables_names:
			var_params = np.prod(np.array(i[1]))
			tot_params += var_params
			print i[0], i[1], var_params
		print "Total number of Trainable Parameters: ", str(tot_params/1000.0)+"K"

	def gen(self, input_img):
		with tf.variable_scope("Generator") as scope:
			conv1 = Conv_2D(input_img, output_chan=3, kernel=[1,1], stride=[1,1], padding="SAME", name="Conv1")
			conv2 = Conv_2D(conv1, output_chan=3, kernel=[3,3], stride=[1,1], padding="SAME", name="Conv2")
			concat_1 = tf.concat([conv1, conv2], axis=3, name="Concat_1")
			conv3 = Conv_2D(concat_1, output_chan=3, kernel=[5,5], stride=[1,1], padding="SAME", name="Conv3")
			concat_2 = tf.concat([conv2, conv3], axis=3, name="Concat_2")
			conv4 = Conv_2D(concat_2, output_chan=3, kernel=[7,7], stride=[1,1], padding="SAME", name="Conv4")
			concat_3 = tf.concat([conv1, conv2, conv3, conv4], axis=3, name="Concat_3")
			conv5 = Conv_2D(concat_3, output_chan=3, kernel=[3,3], stride=[1,1], padding="SAME", name="Conv5_K")
			clearImage = tf.add(tf.multiply(conv5, self.haze_in) - conv5, 1, name="Clear_Image")
			return clearImage
	
	def dis(self, input_img, reuse=False):
		with tf.variable_scope("Discriminator", reuse=reuse) as scope:
			conv1 = Conv_2D(input_img, output_chan=16, kernel=[3,3], stride=[1,1],name="Conv1")
			pool1 = max_pool(conv1, name="pool1")
			conv2 = Conv_2D(pool1, output_chan=32, kernel=[3,3], stride=[1,1], name="Conv2")
			pool2 = max_pool(conv2, name="pool2")
			conv3 = Conv_2D(pool2, output_chan=64, kernel=[3,3], stride=[1,1], name="Conv3")
			pool3 = max_pool(conv3, name="pool3")
			conv4 = Conv_2D(pool3, output_chan=128, kernel=[3,3], stride=[2,2], name="Conv4")
			conv4_reshape = tf.reshape(conv4, shape=[-1, int(np.prod(conv4.get_shape()[1:]))])
			linear1 = Dense(conv4_reshape, output_dim=512, name="dense1")
			linear2 = Dense(linear1, output_dim =1, activation=tf.sigmoid, name="dense2")
			return linear2

	def build_model(self):
		with tf.name_scope("Inputs") as scope:
			self.haze_in = tf.placeholder(tf.float32, shape=[None,240,320,3], name="Haze_Image")
			self.clear_in = tf.placeholder(tf.float32, shape=[None,240,320,3], name="Clear_Image")
			self.train_phase = tf.placeholder(tf.bool, name="is_training")
			hazy_summ = tf.summary.image("Hazy_image", self.haze_in)
			clear_summ = tf.summary.image("clear_in", self.clear_in)

		with tf.name_scope("Model") as scope:
			self.clearImage = self.gen(self.haze_in)
			self.dis_real = self.dis(self.clear_in, reuse=False)
			self.dis_fake = self.dis(self.clearImage, reuse=True)
			clear_image_summ = tf.summary.image("Out_Clear", self.clearImage)
			dis_real_summ = tf.summary.scalar("Dis_Real", tf.reduce_mean(self.dis_real))
			dis_fake_summ = tf.summary.scalar("Dis_Fake", tf.reduce_mean(self.dis_fake))

		with tf.name_scope("Loss") as scope:
			self.dis_loss = -tf.reduce_mean(tf.log(self.dis_real) + tf.log(1-self.dis_fake))
			self.gen_loss = -tf.reduce_mean(tf.log(self.dis_fake))
			self.dis_loss_summ = tf.summary.scalar("Dis_Loss", self.dis_loss)
			self.gen_loss_summ = tf.summary.scalar("Gen_Loss", self.gen_loss)
							 
		with tf.name_scope("Optimizers") as scope:
			train_vars = tf.trainable_variables()
			self.d_vars = [var for var in train_vars if "Discriminator" in var.name]
			self.g_vars = [var for var in train_vars if "Generator" in var.name]

			self.dis_solver = tf.train.AdamOptimizer(learning_rate=1e-05).minimize(self.dis_loss, var_list=self.d_vars)
			self.gen_solver = tf.train.AdamOptimizer(learning_rate=1e-04).minimize(self.gen_loss, var_list=self.g_vars)

		self.merged_summ = tf.summary.merge_all()
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		self.sess = tf.Session(config=config)
		self.train_writer = tf.summary.FileWriter(self.graph_path+'train/')
		self.train_writer.add_graph(self.sess.graph)
		self.val_writer = tf.summary.FileWriter(self.graph_path+'val/')
		self.val_writer.add_graph(self.sess.graph)
		self.saver = tf.train.Saver()
		self.sess.run(tf.global_variables_initializer())
		self._debug_info()

	def train_model(self, train_imgs, val_imgs, learning_rate=1e-5, batch_size=32, epoch_size=50):
		
		print "Training Images: ", train_imgs.shape[0]
		print "Validation Images: ", val_imgs.shape[0]
		print "Learning_rate: ", learning_rate, "Batch_size", batch_size, "Epochs", epoch_size
		raw_input("Training will start above configuration. Press Enter to Start....")
		
		with tf.name_scope("Training") as scope:
			for epoch in range(epoch_size):
				for itr in xrange(0, train_imgs.shape[0]-batch_size, batch_size):
					haze_in = train_imgs[itr:itr+batch_size,0]
					clear_in = train_imgs[itr:itr+batch_size,1]

					dis_in = [self.dis_solver, self.dis_loss, self.merged_summ]
					dis_out = self.sess.run(dis_in, {self.haze_in:haze_in, self.clear_in: clear_in,self.train_phase:True})
					self.train_writer.add_summary(dis_out[2])
					
					for num in range(2):
						gen_in = [self.gen_solver, self.gen_loss, self.merged_summ]
						gen_out = self.sess.run(gen_in, {self.haze_in:haze_in, self.clear_in: clear_in,self.train_phase:True})
						self.train_writer.add_summary(gen_out[2])

					if itr%5==0:
						print "Epoch:", epoch, "Iteration:", itr/batch_size, "Gen Loss:", gen_out[1]
						print "Dis loss:", dis_out[1], "Tot loss:", gen_out[1]+dis_out[1]

						
				for itr in xrange(0, val_imgs.shape[0]-batch_size, batch_size):
					haze_in = val_imgs[itr:itr+batch_size,0]
					clear_in = val_imgs[itr:itr+batch_size,1]

					out = self.sess.run([self.dis_loss,self.gen_loss ,self.merged_summ], {self.haze_in: haze_in, 
												self.clear_in: clear_in, self.train_phase:False})
					self.val_writer.add_summary(out[2])

					print "Epoch: ", epoch, "Iteration: ", itr/batch_size, "Validation Loss: ", out[0]+out[1]

				if epoch%10==0:
					self.saver.save(self.sess, self.save_path+"AOD", global_step=epoch)
					print "Checkpoint saved"

					# a = np.random.randint(1, train_phaseimgs[0].shape[0], 1)
					
					# random_img = train_imgs[0][a]

					# gen_imgs = self.sess.run(self.clearImg, {self.haze_in: random_img[:,0,:,:,:], self.trans_in: train_imgs[1][a], self.train_phase:False})
					# for i,j,k in zip(random_img[:,0,:,:,:], random_img[:,1,:,:,:], gen_imgs):
					# 	stack = np.hstack((i,j,k))
					# 	cv2.imwrite(self.output_path +str(epoch)+"_train_img.jpg", 255.0*stack)

	def test(self, input_imgs, batch_size):
		sess=tf.Session()
		
		saver = tf.train.import_meta_graph(self.save_path+'AOD-50.meta')
		print self.save_path
		saver.restore(sess,tf.train.latest_checkpoint(self.save_path))
		print self.save_path
		graph = tf.get_default_graph()

		x = graph.get_tensor_by_name("Inputs/Haze_Image:0")
		is_train = graph.get_tensor_by_name("Inputs/is_training:0")
		y = graph.get_tensor_by_name("Model/Generator/Clear_Image:0")

		print "Tensor Loaded"
		for itr in xrange(0, input_imgs.shape[0], batch_size):
			if itr+batch_size<=input_imgs.shape[0]:
				end = itr+batch_size
			else:
				end = input_imgs.shape[0]
			input_img = input_imgs[itr:end]
			out = sess.run(y, {x:input_img, is_train:False})
			if itr==0:
				tot_clr = out
			else:
				tot_clr = np.concatenate((tot_clr, out))
		print "Output Shape:", tot_clr.shape
		return tot_clr

		# img_name = glob.glob("/media/mnt/dehaze/*_resize.jpg")
		# for image in img_name:
		# 	img = cv2.imread(image)
		# 	in_img = img.reshape((1, img.shape[0],img.shape[1],img.shape[2]))
		# 	out = sess.run([y, clr_img_clip, clr_img, airlight], {x:in_img/255.0, is_train:False})
		# 	# plt.imshow(out[2][0])
		# 	# plt.show()
		# 	# plt.imshow(out[3][0])
		# 	# plt.show()
		# 	# maps = sess.run(y, {x:in_img/255.0, is_train:False})
		# 	# clear = clearImg(img, maps[0])
		# 	cv2.imwrite(image[:-4]+"_map.jpg", out[0][0]*255.0)
		# 	cv2.imwrite(image[:-4]+"_clear.jpg", out[1][0]*255.0)
