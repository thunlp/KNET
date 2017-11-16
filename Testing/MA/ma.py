import structure as s
import sys
import numpy as np

batch_size = 1000
test_size = 100000
prob = 0.5

'''read in data'''
test_entity = np.load('test_entity.npy')
test_context = np.load('test_context.npy')
test_label = np.load('test_label.npy')

linktest = np.load('linktest.npy')

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
			table = linktest[i*batch_size : (i+1)*batch_size]
		elif version=='miss':
			table = np.ones([batch_size]) - linktest[i*batch_size : (i+1)*batch_size]		
		else:
			table = np.zeros([batch_size])
		
		fdt = s.fdict(i*batch_size, batch_size, test_entity, test_context, test_label, False)

		t = -1
		if version=='person':
			t = 13
		elif version=='organization':
			t = 0
		elif version=='location':
			t = 54
		if t>-1:			
			for j in range(batch_size):			
				if test_label[i*batch_size+j][t]==1:
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
	
	
'''training'''

print('\nStart testing')
s.sess.run(s.initializer)

s.saver.restore(s.sess, '../../Parameter/para_ma/model-3900')


	
test('all')
test('succ')
test('miss')
test('person')
test('organization')
test('location')
