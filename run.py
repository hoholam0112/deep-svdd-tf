import tensorflow as tf 
import numpy as np
import os
import sys
import argparse
import importlib
import models
from utils import VariableManeger
from evaluations import do_roc

# Bulid a compatational graph

def run(args):
	# GPU assignment
	os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
	os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

	# set random seed 
	tf.set_random_seed(args.rd)
	rs = np.random.RandomState(args.rd)

	# get model's hyper-parameters
	nb_epochs = args.nb_epochs # step size for inner updates
	latent_dim = args.latent_dim
	learning_rate = args.lr
	weight_decay_rate = 1 - args.weight_decay_rate
	label = args.label
	batch_size = args.batch_size 

	# Display hyper-parameter information
	print('=========================================================')
	print('Deep SVDD for anomaly detection')
	print('nb_epochs: {}'.format(nb_epochs))
	print('label: {}'.format(label))
	print('learning_rate: {}'.format(learning_rate))
	print('random seed: ', args.rd)	
	print('=========================================================')

	# Get data 
	dataset = importlib.import_module(args.dataset)
	dataset_ = dataset.get_data(label=label, flatten=False, centered=True)

	x_train = dataset_['x_train']
	x_test = dataset_['x_test']
	y_test = dataset_['y_test']
	
	# Build a computational graph
	if args.dataset == 'mnist':
		model_ = models.MnistModel(latent_dim)
	else:
		raise ValueError('Invalid name for dataset')

	encoder = model_.encoder
	decoder = model_.decoder
	input_ph = tf.placeholder(tf.float32, shape = model_.input_shape)

	with tf.variable_scope('encoder'):
		embed = encoder(input_ph, is_training=True, reuse=False)
	with tf.variable_scope('decoder'):
		x_hat = decoder(embed, is_training=True, reuse=False)


	# Center initialization operation
	center = tf.get_variable('center', shape=[latent_dim], trainable=False)
	center_ph = tf.placeholder(tf.float32, shape=center.get_shape(), name='center_ph')
	center_assign_op = tf.assign(center, center_ph)

	# Loss function
	recon_error = tf.reduce_mean(tf.norm(tf.layers.flatten(x_hat - input_ph), axis=1))

	difference = tf.layers.flatten(embed - tf.reshape(center, [1, latent_dim]))
	anomaly_score = tf.norm(difference, axis=1)
	mean_dist = tf.reduce_mean(anomaly_score)
	loss = mean_dist

	# Optimizer
	enc_vars = tf.trainable_variables('encoder')
	dec_vars = tf.trainable_variables('decoder')
	with tf.name_scope('optimizer'):
		train_ae_op = tf.train.AdamOptimizer(10*learning_rate).minimize(recon_error, var_list=enc_vars+dec_vars)
		train_svdd_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=enc_vars)

	# Control variables
	with tf.variable_scope('control_variables'):
		curr_step = tf.get_variable('curr_step', [], dtype=tf.int32, initializer=tf.zeros_initializer)
		plus_curr_step = tf.assign(curr_step, curr_step+1)

	global_init_op = tf.global_variables_initializer()
	optim_init_op = tf.variables_initializer(tf.global_variables('optimizer'))

	with tf.Session() as sess:
		vm = VariableManeger(tf.trainable_variables('decoder'), session=sess, decay_rate=weight_decay_rate)
		sess.run(global_init_op)

		# Saver 
		saver = tf.train.Saver(var_list = tf.global_variables(), max_to_keep = 1)
		save_path = os.path.join('train_logs', args.dataset, str(args.label), args.save_path)
		
		if not os.path.exists(save_path):
			os.mkdir(save_path)

		if args.pretrained:
			print('Restore Model')
			saver.restore(sess, '{}/model.ckpt'.format(save_path))


		# Train the model 
		print('Pretrain model')
		if not args.pretrained:
			for i in range(args.nb_epochs_pre):
				train_loss = train(session=sess, x_data=x_train, batch_size=batch_size, train_op=train_ae_op, 
					loss_op=recon_error, input_ph=input_ph, variable_manager=vm, decay=False)

				if i%1 == 0:
					print('\rstep_%d, train loss: %.4f'%(i, train_loss))


		print('Train model')
		i = sess.run(plus_curr_step)
		while i <= nb_epochs:
			i = sess.run(plus_curr_step)
	
			if i > nb_epochs:
				break
			if i == 1:
				center_init = sess.run(embed, {input_ph:x_train[:batch_size]})
				center_init = np.mean(center_init, axis=1)
				sess.run(center_assign_op, {center_ph:center_init})

			train_loss = train(session=sess, x_data=x_train, batch_size=batch_size, train_op=train_svdd_op, 
				loss_op=loss, input_ph=input_ph, variable_manager=vm, decay=True)

			if i%1 == 0:
				print('\rstep_%d, train loss: %.4f'%(i, train_loss))

			if i%10 == 0:
				saver.save(sess, '{}/model.ckpt'.format(save_path))

			# Evaluate model
			if i%10==0:
				evaluate(session=sess, x_data=x_test, y_data=y_test, batch_size=batch_size, input_ph=input_ph, 
				anomaly_score_op=anomaly_score, file_name=args.save_path, dataset_name=args.dataset, label=args.label)

		if args.test:
			evaluate(session=sess, x_data=x_test, y_data=y_test, batch_size=batch_size, input_ph=input_ph, 
				anomaly_score_op=anomaly_score, file_name=args.save_path, dataset_name=args.dataset, label=args.label)


def train(session, x_data, batch_size, train_op, loss_op, input_ph, variable_manager, decay=False):
	train_loss = 0.0
	nb_batch = int(x_data.shape[0]/batch_size)
	for j in range(nb_batch):
		# show progress
		print("\rComplete: {}%".format(int(100*((j+1)/nb_batch))), end="")
		sys.stdout.flush()
		# run train op
		batch_inputs = x_data[j*batch_size : (j+1)*batch_size] 
		_, train_loss_ = session.run([train_op, loss_op], {input_ph:batch_inputs})
		train_loss += train_loss_

		if decay:
			# weight decay
			variable_manager.weight_decay()

	train_loss /= nb_batch

	return train_loss




def evaluate(session, x_data, y_data, batch_size, input_ph, anomaly_score_op, file_name, dataset_name, label):
	nb_batch = int(x_data.shape[0]/batch_size)
	nb_remainder = x_data.shape[0]%batch_size
	scores = []

	for j in range(nb_batch):
		batch_inputs = x_data[j*batch_size : (j+1)*batch_size]
		scores.append(session.run(anomaly_score_op, {input_ph:batch_inputs}))

	if nb_remainder != 0:
		batch_inputs = x_data[nb_batch*batch_size:]
		scores.append(session.run(anomaly_score_op, {input_ph:batch_inputs}))

	scores = np.concatenate(scores, axis=0)

	roc_auc = do_roc(scores, y_data,
	    file_name=r'{}'.format(file_name),
	    directory=r'results/{}/{}'.format(dataset_name, label))

	print("Testing | ROC AUC = {:.4f}".format(roc_auc))

	return



if __name__ == "__main__":
	def str_to_bool(x):
		if str(x).lower() in ['true', 't']:
			return True
		elif str(x).lower() in ['false', 'f']:
			return False 
		else:
			raise argparse.ArgumentTypeError('Boolean value expected.')

	parser = argparse.ArgumentParser(description='Deep support data description for anomaly detection')
	parser.add_argument('dataset', nargs="?", choices=['mnist'], 
		help='the name of the dataset you want to run the experiments on')
	parser.add_argument('--label', help='label', type=int)
	parser.add_argument('--nb_epochs_pre', help='number of epochs', default=200, type=int)
	parser.add_argument('--nb_epochs', help='number of epochs', default=300, type=int)
	parser.add_argument('--rd', help='random_seed', default=1, type=int)
	parser.add_argument('--gpu', help='gpu_number', default=0, type=int)
	parser.add_argument('--latent_dim', help='latent_dim', default=32, type=int)
	parser.add_argument('--lr', help='learning_rate', default=1e-5, type=float)
	parser.add_argument('--weight_decay_rate', help='weight_decay_rate', default=1e-6, type=float)      
	parser.add_argument('--batch_size', help='batch_size', default=200, type=int) 

	parser.add_argument('--save_path', help='save_path', default='a', type=str)
	parser.add_argument('--test', help='test trained model', default=False, type=str_to_bool)
	parser.add_argument('--pretrained', help='have pretrained model', default=False, type=str_to_bool)

	run(parser.parse_args())


