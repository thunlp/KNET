import tensorflow as tf
import numpy as np
import read as r
from functools import reduce

transe_size = 100
dev = 0.01
lstm_size = 100


'''constructing the NN'''
entity_in = tf.placeholder(tf.float32, [None, r.word_size], name='entityin')

kprob = tf.placeholder(tf.float32, name='kprob')
entity_drop = tf.nn.dropout(entity_in, kprob)

left_in = [ tf.placeholder(tf.float32, [None, r.word_size], name='leftin'+str(i)) for i in range(r.window) ] #from middle to side
right_in = [ tf.placeholder(tf.float32, [None, r.word_size], name='rightin'+str(i)) for i in range(r.window) ]#from middle to side
left_in_rev = [ left_in[r.window-1-i] for i in range(r.window) ]  #from side to middle
right_in_rev = [ right_in[r.window-1-i] for i in range(r.window) ]#from side to middle


left_middle_lstm = tf.nn.rnn_cell.LSTMCell(lstm_size)
right_middle_lstm = tf.nn.rnn_cell.LSTMCell(lstm_size)
left_side_lstm = tf.nn.rnn_cell.LSTMCell(lstm_size)
right_side_lstm = tf.nn.rnn_cell.LSTMCell(lstm_size)

with tf.variable_scope('root'):
	
	with tf.variable_scope('lstm'):
		left_out_rev, state0 = tf.nn.rnn(left_middle_lstm, left_in_rev, dtype=tf.float32)   #from side to middle
	
	with tf.variable_scope('lstm', reuse=True):
		right_out_rev, state1 = tf.nn.rnn(right_middle_lstm, right_in_rev, dtype=tf.float32)#from side to middle
		left_out, state2 = tf.nn.rnn(left_side_lstm, left_in, dtype=tf.float32)   #from middle to side
		right_out, state3 = tf.nn.rnn(right_side_lstm, right_in, dtype=tf.float32)#from middle to side



#attention layer

left_att_in = [ tf.concat(1, [left_out[i], left_out_rev[r.window-1-i]]) for i in range(r.window) ]   #left then right
right_att_in = [ tf.concat(1, [right_out[i], right_out_rev[r.window-1-i]]) for i in range(r.window) ]#right then left
#both side then middle
one = tf.placeholder(tf.bool)
query = tf.cond(one, lambda:tf.ones([tf.shape(entity_in)[0], r.word_size]), lambda:entity_in)

A = tf.Variable(tf.random_normal([lstm_size*2, r.word_size], mean=0, stddev=dev))

left_att = [ tf.pow(tf.reduce_sum(tf.matmul(left_att_in[i], A) * query, [1], keep_dims=True),2)\
	for i in range(r.window) ]
right_att = [ tf.pow(tf.reduce_sum(tf.matmul(right_att_in[i], A) *  query, [1], keep_dims=True),2)\
	for i in range(r.window) ]


left_weighted = reduce(tf.add, [ left_att_in[i]*left_att[i] for i in range(r.window) ])
right_weighted = reduce(tf.add, [ right_att_in[i]*right_att[i] for i in range(r.window) ])

left_all = reduce(tf.add, [ left_att[i] for i in range(r.window) ])
right_all = reduce(tf.add, [ right_att[i] for i in range(r.window) ])

context_in = tf.concat(1, [left_weighted/left_all, right_weighted/right_all])


x = tf.concat(1, [entity_drop, context_in])
W = tf.Variable(tf.random_normal([r.word_size+lstm_size*4, transe_size], stddev=dev), name='W')
y = tf.nn.tanh(tf.matmul(x, W)) 
T = tf.Variable(tf.random_normal([transe_size, r.type_size], stddev=dev), name='T')
t = tf.nn.sigmoid(tf.matmul(y, T)) #learned type prob
t_ = tf.placeholder(tf.float32, [None, r.type_size], name='t_') #real type



loss = - tf.reduce_sum(t_*tf.log(t+1e-10) + (1-t_)*tf.log(1-t+1e-10))

train = tf.train.AdamOptimizer(0.005).minimize(loss)

opt1 = tf.train.AdamOptimizer(0.005)
grad1 = opt1.compute_gradients(loss)
train1 = opt1.apply_gradients(grad1)


sess = tf.Session()
initializer = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=100)


'''the function of obtaining the result'''
def guess(y, y_, sess, fd, ctb, th=0.5):
	showy = sess.run(y, feed_dict=fd)
	showy_ = fd[y_]
	h = showy > th
	strict = 0
	lma_p = 0
	lma_r = 0
	
	for i in range(np.shape(h)[0]):
		if ctb[i]==1:
			if np.sum(h[i, :])==0:
				h[i, np.argmax(showy[i, :])] = 1
			
			#strict
			count = True
			for j in range(r.type_size):
				if h[i, j] != showy_[i, j]:
					count = False
					break
			if count:
				strict += 1
			
			#loose macro
			tp = float(np.sum(np.logical_and(h[i], showy_[i])))
			fp = float(np.sum(h[i]))
			tn = float(np.sum(showy_[i]))
			lma_p += tp/fp
			if tn!=0:
				lma_r += tp/tn
		
	
	#loose micro
	table = np.transpose(np.tile(ctb, [r.type_size, 1]))
	true_pos = float(np.sum(np.logical_and(table, np.logical_and(h, showy_))))
	false_pos = float(np.sum(np.logical_and(table, h)))
	true_neg = float(np.sum(np.logical_and(table, showy_)))
	
	effect = float(np.sum(ctb))
	return (float(strict), lma_p, lma_r, true_pos, false_pos, true_neg, effect)


'''the function of constructing a feed_dict from dataset'''
def fdict(now, size, _entity, _context, _label, ones):
	fd = {}
	
	ent = np.zeros([size, r.word_size])
	lab = np.zeros([size, r.type_size])
	for i in range(size):
		vec = np.zeros([r.word_size])
		l = len(_entity[now+i])
		for j in range(l):
			vec += r.dic( _entity[now+i][j] )
		ent[i] = vec/l
		lab[i] = _label[now+i]
	fd[entity_in] = ent
	fd[t_] = lab
	fd[one] = ones
	
	for j in range(r.window):# window3 j0 jj2; j1 jj1; j2 jj0;
		jj = r.window - j -1 # reversion happens here
		left_con = np.zeros([size, r.word_size])
		right_con = np.zeros([size, r.word_size])
		for i in range(size):
			left_con[i, :] = r.dic( _context[now+i][2*jj] )
			right_con[i, :] = r.dic( _context[now+i][2*jj+1] )
		fd[left_in[j]] = left_con
		fd[right_in[j]] = right_con
		
	return fd
	
