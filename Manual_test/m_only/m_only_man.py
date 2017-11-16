import structure as s
import numpy as np
import sys


batch_size = 100
test_size = 100

'''read in data'''
embedding = np.load('embedding.npy')

train_entity = np.load('train_entity.npy')
train_context = np.load('/rain_context.npy')
train_label = np.load('train_label.npy')
train_fbid = np.load('train_fbid.npy')

manual_entity = np.load('manual_entity.npy')
manual_context = np.load('manual_context.npy')
manual_label = np.load('manual_label.npy')
manual_fbid = np.load('/manual_fbid.npy')



linkman = np.load('linkmanual.npy')

def test(version):
	true_pos = 0
	false_pos = 0
	true_neg = 0	
	strict = 0
	lma_p = 0
	lma_r = 0
	effect = 0
	#organization 0 person 13 location 54

	
	for i in range(int(test_size/batch_size)):		

		if version=='all':
			table = np.ones([batch_size])
		elif version=='succ':
			table = linkman[i*batch_size : (i+1)*batch_size]
		elif version=='miss':
			table = np.ones([batch_size]) - linkman[i*batch_size : (i+1)*batch_size]		
		else:
			table = np.zeros([batch_size])
		
		fdt = s.tfdict(i*batch_size, batch_size, 1, manual_entity, manual_context, manual_label, embedding)
		
		t = -1
		if version=='person':
			t = 13
		elif version=='organization':
			t = 0
		elif version=='location':
			t = 54
		if t>-1:			
			for j in range(batch_size):			
				if manual_label[i*batch_size+j][t]==1:
					table[j] = 1
					
		fdt[s.kprob] = 1.0
		
		
		result = s.guess(s.t, s.t_, s.sess, fdt, table)
		
		strict += result[0]
		lma_p += result[1]
		lma_r += result[2]
		true_pos += result[3]
		false_pos += result[4]
		true_neg += result[5]
		effect += result[6]
		
					

	precision = true_pos / false_pos
	recall = true_pos / true_neg
	print(version)
	print('strict: %f' %(strict/effect))
	print('loose-macro: %f %f %f'   %(lma_p/effect, lma_r/effect, (2*lma_p*lma_r)/(lma_p+lma_r)/effect))
	print('loose-micro: %f %f %f\n\n' %(precision, recall, (precision*recall*2)/(precision+recall)))


s.sess.run(s.initializer)
s.new_saver.restore(s.sess, 'para/model-5600')

print('start training')

for i in range(100):
	fd = s.fdict(i*batch_size, batch_size, 1, train_entity, train_context, train_label, train_fbid, embedding)
	fd[s.kprob] = 0.5
	s.sess.run(s.train1, feed_dict=fd)

print('Start testing')


	
test('all')
test('succ')
test('miss')
test('person')
test('organization')
test('location')
