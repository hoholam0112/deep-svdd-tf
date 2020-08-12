import tensorflow as tf 
import numpy as np 

class VariableManeger():
	def __init__(self, variable_list, session, decay_rate=1):
		self.variable_list = variable_list
		self.sess = session
		self.placeholder_list = [tf.placeholder(var.dtype.base_dtype, shape=var.get_shape()) for var in self.variable_list]
		self.decay_rate = decay_rate

		# computational graph for update
		assign_op_list = [tf.assign(var, ph) for var, ph in zip(variable_list, self.placeholder_list)]
		self.assign_op = tf.group(*assign_op_list)

		# computational graph for weight decay
		decay_ops = [tf.assign(var, var*self.decay_rate) for var in self.variable_list]
		self.decay_op = tf.group(*decay_ops)

	def import_variables(self, new_variable_list):
		# Update parameters
		return self.sess.run(self.assign_op, dict(zip(self.placeholder_list, new_variable_list)))

	def export_variables(self):
		return self.sess.run(self.variable_list)

	def weight_decay(self):
	    return self.sess.run(self.decay_op)

def interpolate_vars(old_var_list, new_var_list, epsilon):
    # Interpolate between two sequences of variables.
    return add_vars(old_var_list, scale_vars(subtract_vars(new_var_list, old_var_list), epsilon))

def average_vars(var_lists):
    # Average a sequence of variable sequences.
    res = []
    for variables in zip(*var_lists):
        res.append(np.mean(variables, axis=0))
    return res

def subtract_vars(var_list1, var_list2):
    # Subtract one variable sequence from another.
    return [v1 - v2 for v1, v2 in zip(var_list1, var_list2)]

def add_vars(var_list1, var_list2):
    # Add two variable sequences.
    return [v1 + v2 for v1, v2 in zip(var_list1, var_list2)]

def scale_vars(var_list, scale):
    # Scale a variable sequence.
    return [v * scale for v in var_list]

def sum_vars(var_lists):
	res = []
	for variables in zip(*var_lists):
		res.append(np.sum(variables, axis=0))

	return res





if __name__ == '__main__':
	# with tf.variable_scope('scope1'):
	# 	a = tf.get_variable('a', [2,2])
	# 	b = tf.get_variable('b', [10])
	# 	c = tf.get_variable('c', [])

	# with tf.variable_scope('scope2'):
	# 	a = tf.get_variable('a', [2,2])
	# 	b = tf.get_variable('b', [10])
	# 	c = tf.get_variable('c', [])


	# var_list1 = tf.global_variables('scope1')
	# var_list2 =tf.global_variables('scope2')
	# init_op = tf.global_variables_initializer()

	# with tf.Session() as sess:
	# 	sess.run(init_op)
	# 	var_man1 = VariableManeger(var_list1, sess)
	# 	var_man2 = VariableManeger(var_list2, sess)

	# 	var1 = var_man1.export_variables()
	# 	var2 = var_man2.export_variables()

	# 	print(var1)
	# 	print(var2)

	# 	var_man1.import_variables(var2)

	# 	print(var_man1.export_variables())
	# 	print(var_man2.export_variables())



	a= tf.get_variable('a' , [] , initializer=tf.constant_initializer(1))
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		vm = VariableManeger([a], sess, 0.9)

		for _ in range(10):
			print(vm.export_variables())
			vm.weight_decay()
			print(vm.export_variables())
