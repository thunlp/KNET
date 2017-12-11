import tensorflow as tf
from functools import reduce
import numpy as np
import util
import operator



class Model(object):

    def __init__(self, name):
        self.type_size = 74
        self.word_size = 300
        self.lstm_size = 100
        self.transe_size = 100
        self.dev = 0.01
        self.hidden_layer = 50
        self.window = 15
        self.scope = "root_train_second" if name=="KA+D" else "root"

        self.predict()

    def entity(self):
        self.entity_in = tf.placeholder(tf.float32, [None, self.word_size])
        self.kprob = tf.placeholder(tf.float32)
        entity_drop = tf.nn.dropout(self.entity_in, self.kprob)
        return entity_drop

    def attention(self):
        # this function will be overrided by derived classes
        return None

    def context(self):
        #from middle to side
        self.left_in = [tf.placeholder(tf.float32, [None, self.word_size]) \
            for _ in range(self.window)] 
        self.right_in = [tf.placeholder(tf.float32, [None, self.word_size]) \
            for _ in range(self.window)]

        #from side to middle
        left_in_rev = [self.left_in[self.window-1-i] for i in range(self.window)] 
        right_in_rev = [self.right_in[self.window-1-i] for i in range(self.window)]


        left_middle_lstm = tf.nn.rnn_cell.LSTMCell(self.lstm_size)
        right_middle_lstm = tf.nn.rnn_cell.LSTMCell(self.lstm_size)
        left_side_lstm = tf.nn.rnn_cell.LSTMCell(self.lstm_size)
        right_side_lstm = tf.nn.rnn_cell.LSTMCell(self.lstm_size)

        with tf.variable_scope(self.scope):
            with tf.variable_scope('lstm'):
                #from side to middle
                left_out_rev, _ = tf.nn.rnn(left_middle_lstm, left_in_rev, dtype=tf.float32)
            with tf.variable_scope('lstm', reuse=True):
                #from side to middle
                right_out_rev, _ = tf.nn.rnn(right_middle_lstm, right_in_rev, dtype=tf.float32)
                
                #from middle to side
                left_out, _ = tf.nn.rnn(left_side_lstm, self.left_in, dtype=tf.float32)
                right_out, _ = tf.nn.rnn(right_side_lstm, self.right_in, dtype=tf.float32)

        left_att_in = [tf.concat(1, [left_out[i], left_out_rev[self.window-1-i]]) \
            for i in range(self.window)]
        right_att_in = [tf.concat(1, [right_out[i], right_out_rev[self.window-1-i]]) \
            for i in range(self.window)]

        left_att, right_att = self.attention(left_att_in, right_att_in, left_in_rev, right_in_rev)

        left_weighted = reduce(tf.add, [left_att_in[i]*left_att[i] for i in range(self.window)])
        right_weighted = reduce(tf.add, [right_att_in[i]*right_att[i] for i in range(self.window)])

        left_all = reduce(tf.add, [ left_att[i] for i in range(self.window) ])
        right_all = reduce(tf.add, [ right_att[i] for i in range(self.window) ])

        return tf.concat(1, [left_weighted/left_all, right_weighted/right_all])

    def predict(self):
        # this function will be overrided by derived classes
        return None

    def fdict(self, w2v, now, size, interval, _entity, _context, _label, _fbid, _embedding):
        # this function will be overrided by derived classes
        return None



class SA(Model):

    def attention(self, left_att_in, right_att_in, left_in_rev, right_in_rev):
        W1 = tf.Variable(tf.random_normal([self.lstm_size*2, self.hidden_layer], stddev=self.dev))
        W2 = tf.Variable(tf.random_normal([self.hidden_layer, 1], stddev=self.dev))

        left_att = [tf.exp(tf.matmul(tf.tanh(tf.matmul(left_att_in[i], W1)), W2)) \
            for i in range(self.window)]
        right_att = [tf.exp(tf.matmul(tf.tanh(tf.matmul(right_att_in[i], W1)), W2)) \
            for i in range(self.window)]

        return (left_att, right_att)


    def predict(self):
        x = tf.concat(1, [self.entity(), self.context()])

        W = tf.Variable(tf.random_normal([self.word_size+self.lstm_size*4, self.type_size], \
                stddev=self.dev))
        self.t = tf.nn.sigmoid(tf.matmul(x, W))
        self.t_ = tf.placeholder(tf.float32, [None, self.type_size])

        self.loss = -tf.reduce_sum(self.t_*tf.log(self.t+1e-10)) \
                    -tf.reduce_sum((1-self.t_)*tf.log(1-self.t+1e-10))
        self.train = tf.train.AdamOptimizer(0.005).minimize(self.loss)
        self.saver = tf.train.Saver(max_to_keep=100)
        self.initializer = tf.global_variables_initializer()

    def fdict(self, w2v, now, size, interval, _entity, _context, _label, _fbid, _embedding, _test):
        fd = {}
        new_size = int(size/interval)
        
        ent = np.zeros([new_size, self.word_size])
        lab = np.zeros([new_size, self.type_size])
        for i in range(new_size):
            vec = np.zeros([self.word_size])
            l = len(_entity[now+i*interval])
            for j in range(l):
                vec += util.dic(w2v, _entity[now+i*interval][j])
            ent[i] = vec/l
            lab[i] = _label[now+i*interval]
        fd[self.entity_in] = ent
        fd[self.t_] = lab
        
        for j in range(self.window):# window3 j0 jj2; j1 jj1; j2 jj0;
            left_con = np.zeros([new_size, self.word_size])
            right_con = np.zeros([new_size, self.word_size])
            for i in range(new_size):
                left_con[i, :] = util.dic(w2v, _context[now+i*interval][2*j])
                right_con[i, :] = util.dic(w2v, _context[now+i*interval][2*j+1])
            fd[self.left_in[j]] = left_con
            fd[self.right_in[j]] = right_con
            
        return fd


class KA_D(Model):

    def __init__(self, name):
        Model.__init__(self, name)
        self.disamb = util.build_disamb("data/new_disamb")

    def mag(self, matrix):
        return tf.reduce_sum(tf.pow(matrix, 2))

    def attention(self, left_att_in, right_att_in, left_in_rev, right_in_rev):
        self.middle = 200
        self.max_candidate = 20
        self.disamb_in = tf.placeholder(tf.int32, [None, self.max_candidate], name='disamb_in')
        self.embedding = tf.placeholder(tf.float32, [14951, self.transe_size], name='embedding')

        left_query_lstm = tf.nn.rnn_cell.LSTMCell(self.lstm_size)
        right_query_lstm = tf.nn.rnn_cell.LSTMCell(self.lstm_size)

        with tf.variable_scope(self.scope):
            with tf.variable_scope('query'):
                left_query_out, _ = tf.nn.rnn(left_query_lstm, left_in_rev, dtype=tf.float32)
            with tf.variable_scope('query', reuse=True):
                right_query_out, _ = tf.nn.rnn(right_query_lstm, right_in_rev, dtype=tf.float32)

        #special part from attention query
        query_in = tf.concat(1, [self.entity_in, left_query_out[-1], right_query_out[-1]])
        Wq1 = tf.Variable(tf.random_normal([self.word_size+2*self.lstm_size, self.middle], \
            stddev=self.dev))
        Wq2 = tf.Variable(tf.random_normal([self.middle, self.transe_size], stddev=self.dev))
        self.query = tf.tanh(tf.matmul(tf.tanh(tf.matmul(query_in, Wq1)), Wq2))
        self.query_ = tf.placeholder(tf.float32, [None, self.transe_size]) #real FB representation

        #choose the most likely embedding
        expand = tf.gather(self.embedding, self.disamb_in)
        multi = tf.transpose(tf.pack([self.query]*self.max_candidate), perm=[1,0,2])
        diff = tf.reduce_sum(tf.pow(expand-multi,2), 2)

        smallladder = [ [i] for i in range(1000) ]
        ladder = tf.constant(smallladder, dtype=tf.int64)
        DIFF = tf.expand_dims(tf.argmin(diff, 1), 1)

        #which embedding to choose
        choice = tf.gather_nd(self.disamb_in, tf.concat(1, [ladder, DIFF]))
        

        self.sh = tf.placeholder(tf.float32)
        miss = tf.logical_not(tf.logical_or(\
            tf.equal(self.disamb_in[:,1], 0), \
            tf.less(tf.reduce_min(diff, 1), self.sh))) # should be false for training

        temp_query = tf.gather(self.embedding, choice)
        real_query = tf.select(tf.logical_or(miss, tf.equal(choice, [0]*1000)), \
            self.query, temp_query)

        self.A = tf.Variable(tf.random_normal([self.lstm_size*2, self.transe_size], \
            mean=0, stddev=self.dev))
        self.test = tf.placeholder(tf.bool, [1000])
        Q = tf.select(self.test, real_query, self.query_)

        left_att = [tf.pow(tf.reduce_sum(tf.matmul(left_att_in[i], self.A) * Q, \
            [1], keep_dims=True),2)\
            for i in range(self.window)]
        right_att = [tf.pow(tf.reduce_sum(tf.matmul(right_att_in[i], self.A) * Q, \
            [1], keep_dims=True),2)\
            for i in range(self.window)]

        return (left_att, right_att)

    def predict(self):
        x = tf.concat(1, [self.entity(), self.context()])
        W = tf.Variable(tf.random_normal([self.word_size+self.lstm_size*4, self.transe_size], \
            stddev=self.dev), name='W')
        T = tf.Variable(tf.random_normal([self.transe_size, self.type_size], \
            stddev=self.dev), name='T')
        y = tf.nn.tanh(tf.matmul(x, W)) 
        self.t = tf.nn.sigmoid(tf.matmul(y, T))
        self.t_ = tf.placeholder(tf.float32, [None, self.type_size])


        self.loss1 = - tf.reduce_sum(self.t_*tf.log(self.t+1e-5) \
            + (1-self.t_)*tf.log(1-self.t+1e-5)) \
            + 0.1* (self.mag(self.A)+self.mag(W)+self.mag(T))
        opt1 = tf.train.AdamOptimizer(0.005)
        grad1 = opt1.compute_gradients(self.loss1)
        self.train1 = opt1.apply_gradients(grad1)

        self.loss2 = tf.reduce_sum(tf.pow(self.query-self.query_, 2))
        opt2 = tf.train.AdamOptimizer(0.005)
        grad2 = opt2.compute_gradients(self.loss2)
        self.train2 = opt2.apply_gradients(grad2)


        self.saver = tf.train.Saver(max_to_keep=100)
        self.initializer = tf.global_variables_initializer()


    def fdict(self, w2v, now, size, interval, _entity, _context, _label, _fbid, _embedding, _test):
        if _test:
            fd = {self.test:[True]*1000, self.sh:0.55}
        else:
            fd = {self.test:[False]*1000, self.sh:9999}

        new_size = int(size/interval)
        ent = np.zeros([new_size, self.word_size])
        lab = np.zeros([new_size, self.type_size])

        if not _test:
            fd[self.embedding] = _embedding
            fd[self.disamb_in] = [ [1] * self.max_candidate ] * 1000
            
            fbrepr = np.zeros([new_size, self.transe_size])
            for i in range(new_size):
                vec = np.zeros([self.word_size])
                l = len(_entity[now+i*interval])
                for j in range(l):
                    vec += util.dic(w2v, _entity[now+i*interval][j])
                ent[i] = vec/l
                lab[i] = _label[now+i*interval]
                fbrepr[i] = _embedding[_fbid[now+i*interval]]
            fd[self.entity_in] = ent
            fd[self.t_] = lab
            fd[self.query_] = fbrepr

        else:
            exclude = ['The','of','and','on','at','in','the',\
                       'to','for','or','by','al','St','is','an']
            newemb = _embedding
            newemb[0] = np.ones([self.transe_size])*10
            fd[self.embedding] = newemb
            disa_ = list(self.disamb)
            new_size = int(size/interval)
            
            di = [ [] for _ in range(new_size) ]
            for i in range(new_size):
                vec = np.zeros([self.word_size])
                l = len(_entity[now+i*interval])
                disa = {}
                for j in range(l):
                    vec += util.dic(w2v, _entity[now+i*interval][j])
                    for d in disa_:
                        if      (_entity[now+i*interval][j] in d) \
                            and (len(_entity[now+i*interval][j])>1) \
                            and (_entity[now+i*interval][j] not in exclude):
                                if d in disa:
                                    disa[d] +=1
                                else:
                                    disa[d] = 1
                sorted_disa = sorted(disa.items(), key=operator.itemgetter(1), reverse=True)
                if disa:
                    max = sorted_disa[0][1]
                ent[i] = vec/l
                lab[i] = _label[now+i*interval]
                for so in sorted_disa:
                    if so[1]==max or so[1]==max-1:
                        di[i] += self.disamb[so[0]]
                di[i] += [0] * (self.max_candidate-len(di[i]))
                di[i] = di[i][:self.max_candidate]
            fd[self.entity_in] = ent
            fd[self.t_] = lab
            fd[self.disamb_in] = di
            fd[self.query_] = [ [0.0] * 100 ] *1000
        

        for j in range(self.window):# window3 j0 jj2; j1 jj1; j2 jj0;
            jj = self.window - j -1 # reversion happens here
            left_con = np.zeros([new_size, self.word_size])
            right_con = np.zeros([new_size, self.word_size])
            for i in range(new_size):
                left_con[i, :] = util.dic(w2v, _context[now+i*interval][2*jj])
                right_con[i, :] = util.dic(w2v, _context[now+i*interval][2*jj+1])
            fd[self.left_in[j]] = left_con
            fd[self.right_in[j]] = right_con
            
        return fd