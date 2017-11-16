import structure as s
import numpy as np

batch_size = 1000
train_size = 1000000
test_size = 100000
train_num = 6000
check_num = 100
prob = 0.5

'''read in data'''
embedding = np.load('embedding.npy')

train_entity = np.load('train_entity.npy')
train_context = np.load('train_context.npy')
train_label = np.load('train_label.npy')
train_fbid = np.load('train_fbid.npy')


valid_entity = np.load('valid_entity.npy')
valid_context = np.load('valid_context.npy')
valid_label = np.load('valid_label.npy')
valid_fbid = np.load('valid_fbid.npy')

test_entity = np.load('test_entity.npy')
test_context = np.load('test_context.npy')
test_label = np.load('test_label.npy')
test_fbid = np.load('test_fbid.npy')



fdw = s.fdict(0, train_size, 1000,\
	train_entity, train_context, train_label, train_fbid, embedding)
fdw[s.kprob] = prob




	

def test(n, version):
	true_pos = 0
	false_pos = 0
	true_neg = 0

	unchanged = 0
	correct = 0
	wrong = 0
	
	for i in range(int(test_size/batch_size)):
		if version=='test':
			fdt = s.tfdict(i*batch_size, batch_size, 1, test_entity, test_context, test_label, embedding)
		else:
			fdt = s.tfdict(i*batch_size, batch_size, 1, valid_entity, valid_context, valid_label, embedding)
		fdt[s.kprob] = 1.0
		result = s.guess(s.t, s.t_, s.sess, fdt)
		
		true_pos += result[3]
		false_pos += result[4]
		true_neg += result[5]
		if True:
			schoice = s.sess.run(s.choice, feed_dict=fdt)
			if version=='test':
				for j in range(1000):
					if schoice[j]==0:
						unchanged += 1
					elif schoice[j]==test_fbid[i*batch_size+j]:
						correct += 1
					else:
						wrong += 1
			else:#valid
				for j in range(1000):
					if schoice[j]==0:
						unchanged += 1
					elif schoice[j]==valid_fbid[i*batch_size+j]:
						correct += 1
					else:
						wrong += 1
						


	precision = true_pos / float(false_pos)
	recall = true_pos / true_neg
	print(version)
	print('%d %f %f %f' %(n, precision, recall, (precision*recall*2)/(precision+recall)))
	print('u c w %d %d %d' %(unchanged, correct, wrong))

'''training'''

print('\nStart training')
s.sess.run(s.initializer)
Now = 0

for i in range(train_num):
	fd = s.fdict(Now, batch_size, 1,\
		train_entity, train_context, train_label, train_fbid, embedding)
	fd[s.kprob] = prob
	Now += batch_size
	if Now>=train_size:
		Now = 0
	
	
	if i%check_num==0 or i==train_num-1:
		result = s.guess(s.t, s.t_, s.sess, fd)
		print('%d %f %f %f\t%f\t%f' %(i, result[0], result[1], result[2], s.sess.run(s.loss1, feed_dict=fd), s.sess.run(s.loss2, feed_dict=fd)))
		result = s.guess(s.t, s.t_, s.sess, fdw)
		print('%d %f %f %f\t%f\t%f' %(i, result[0], result[1], result[2], s.sess.run(s.loss1, feed_dict=fdw), s.sess.run(s.loss2, feed_dict=fdw)))
		test(i, 'valid')
		test(i, 'test')

		s.new_saver.save(s.sess, '../nouse_para/model', global_step=i)
		
		print('')
	s.sess.run(s.train1, feed_dict=fd)
	s.sess.run(s.train2, feed_dict=fd)
