import numpy as np

'''''''''''hyperparameter'''''''''''''''''
type_size = 74
typeind_size = 5
train_size = 1000000 #1 million
word_size = 300
window = 15
test_size = 100000 #0.1million


'''''''''''input type'''''''''''''''''
t2n = {} #convert type as string to number
n2t = {}
with open('type.stat', 'r') as f1:
	for i in range(type_size):
		s = f1.readline()[:-1]
		a = s.split('\t')
		t2n[a[0]] = int(a[1])
		n2t[int(a[1])] = a[0]

pt = {}
with open('type.parent', 'r') as f2:
	for i in range(typeind_size):
		s = f2.readline()
		a = s.split('\t')
		pt[a[0][1:]] = int(a[1])
	
t2t = {} #convert subtype to its parent type
for ty in t2n:
	parent = ty.split('/')[1]
	if parent in pt:
		t2t[t2n[ty]] = pt[parent]

print('end of type input')


'''''''''''input word vectors'''''''''''''''''
w2v = {}
vocab = 0 #size of vocabulary
with open('glove.840B.300d.txt', 'r') as f3:
	while True:
		s = f3.readline()
		vec = np.zeros([word_size])
		space = s.find(' ')
		word = s[:space]
		s = s[space+1:]
		for i in range(word_size-1):
			space = s.find(' ')
			vec[i] = float(s[:space])
			s = s[space+1:]
		vec[word_size-1] = float(s)
		w2v[word] = vec
		vocab += 1
		
		if word=='unk':
			print('unk found!')
			print('vocabulary size: %d' %vocab)
			unk = w2v['unk']
			break
		'''if vocab==100:
			print('halts at vocab==100')
			unk = np.zeros([word_size])
			w2v['unk'] = unk
			break'''
print('end of word2vec input')

def dic(s): #looking up the dictionary
	if s in w2v:
		return w2v[s]
	else:
		return unk


'''''''''''input training and test data'''''''''''''''''
def parse(fin, _entity, _context, _label, _fbid, size):
	with open(fin, 'r') as f:
		for i in range(size):
			entity = []
			token = []
	
			#parsing entity mention
			f.readline() #'/0'
			s = f.readline()[:-1] #'Greek'
			while s!='/1/': #read in all the words in entity mention
				entity.append(s)
				s = f.readline()[:-1] #'/1/'
			_entity[i] = entity
			
			#parsing head
			head = int(f.readline()[:-1])#'3'
			
			#parsing context
			f.readline() #'/2/'
			s = f.readline()[:-1] #'Leros'
			while s!='/3/':
				token.append(s)
				s = f.readline()[:-1] #'is'~'/3/'
			
			for j in range(window):
				k = j+1
				if 0<=head-k and head-k<len(token):
					_context[i].append(token[head-k])
				else:
					_context[i].append('unk')
					
				tail = head+len(entity)
				if tail+j<len(token):
					_context[i].append(token[tail+j])
				else:
					_context[i].append('unk')
					
			#parsing label
			s = f.readline()[:-1] #'location country'
			while s!='/4/':
				if s in t2n:
					t = t2n[s]
					_label[i, t] = 1
					if t in t2t: #included the parent type
						_label[i, t2t[t]] = 1
				else:
					print('missing type: %s' %s)
				s = f.readline()[:-1] #'location ...'~'/4/'
			
			#parsing FBID
			s = f.readline()[:-1]
			_fbid[i] = int(s.split('\t')[1]) #035qy   13500
			#fbid starts from 0
						
			#end of parsing of one training data
	print('end of parsing\t' + fin)
	
	

'''''''''''''input transE embedding'''''''''''''''
def emb(file, embedding):
	with open(file, 'r') as f:
		num = 0
		for s in f:
			a = s.split()
			embedding[num, :] = a
			num += 1


'''''''''''test of readin itself''''''''''''''
def checking():
	s = input()
	while s!='':
		if s in w2v:
			print(w2v[s])
		else:
			print('unk')
			print(unk)
		s = input()
	return 0'''