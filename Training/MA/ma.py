import structure as s
import sys
import numpy as np

batch_size = 1000
train_num = 4000
check_num = 100
prob = 0.5

'''read in data'''
train_entity = np.load('train_entity.npy')
train_context = np.load('train_context.npy')
train_label = np.load('train_label.npy')


valid_entity = np.load('valid_entity.npy')
valid_context = np.load('valid_context.npy')
valid_label = np.load('valid_label.npy')

test_entity = np.load('test_entity.npy')
test_context = np.load('test_context.npy')
test_label = np.load('test_label.npy')

def test(n, version):
	true_pos = 0
	false_pos = 0
	true_neg = 0

	for i in range(100):
		if version=='test':
			fdt = s.fdict(i*batch_size, batch_size, test_entity, test_context, test_label, False)
		else:
			fdt = s.fdict(i*batch_size, batch_size, valid_entity, valid_context, valid_label, False)
		fdt[s.kprob] = 1.0
		result = s.guess(s.t, s.t_, s.sess, fdt)
		
		true_pos += result[0]
		false_pos += result[1]
		true_neg += result[2]


	precision = true_pos / false_pos
	recall = true_pos / true_neg
	print(version)
	print('%d %f %f %f' %(n, precision, recall, (precision*recall*2)/(precision+recall)))

'''training'''

print('\nStart training')
s.sess.run(s.initializer)

#delete the following comment mark at the end
for i in range(train_num):
	if i<3:#000:
		fd = s.fdict((i%1000)*1000, batch_size, train_entity, train_context, train_label, True)
		fd[s.kprob] = 0.5
		s.sess.run(s.train, feed_dict=fd)
	else:
		fd = s.fdict((i%1000)*1000, batch_size, train_entity, train_context, train_label, False)
		fd[s.kprob] = 0.5
		s.sess.run(s.train1, feed_dict=fd)
	
	
	if i%4==0:#check_num==0:
		test(i, 'test')
		test(i, 'valid')
		s.saver.save(s.sess, '../nouse_para/model', global_step=i)
		print('')
		