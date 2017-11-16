import structure as s
import numpy as np

batch_size = 1000
train_num = 9000
check_num = 100
prob = 0.5
train_size = 1000000
test_size = 100000

'''read in data'''
emb = np.load('embedding.npy')

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


def test(n, version):
	true_pos = 0
	false_pos = 0
	true_neg = 0

	for i in range(100):
		if version=='test':
			fdt = s.fdict(i*batch_size, batch_size, test_entity, test_context, test_label, test_fbid, emb, False)
		else:
			fdt = s.fdict(i*batch_size, batch_size, valid_entity, valid_context, valid_label, valid_fbid, emb, False)
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

for i in range(train_num):
	if i<6000:
		fd = s.fdict((i%1000)*1000, batch_size, train_entity, train_context, train_label, train_fbid, emb, True)
		fd[s.kprob] = 0.5
		s.sess.run(s.train, feed_dict=fd)
		s.sess.run(s.train2, feed_dict=fd)
	else:
		fd = s.fdict((i%1000)*1000, batch_size, train_entity, train_context, train_label, train_fbid, emb, False)
		fd[s.kprob] = 0.5
		s.sess.run(s.train1, feed_dict=fd)
	
	if i%check_num==0 or i==train_num-1:
		test(i, 'test')
		test(i, 'valid')
		s.saver.save(s.sess, '../nouse_para/model', global_step=i)
		print('')
		