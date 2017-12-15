import sys
import numpy as np
import model
import util
import tensorflow as tf


batch_size = 1000

# load data
datadir = sys.argv[1]
wv_file = sys.argv[2]

embedding = np.load(datadir+'/embedding.npy')

# train_entity = np.load(datadir+'/train_entity.npy')
# train_context = np.load(datadir+'/train_context.npy')
# train_label = np.load(datadir+'/train_label.npy')
# train_fbid = np.load(datadir+'/train_fbid.npy')

# valid_entity = np.load(datadir+'/valid_entity.npy')
# valid_context = np.load(datadir+'/valid_context.npy')
# valid_label = np.load(datadir+'/valid_label.npy')
# valid_fbid = np.load(datadir+'/valid_fbid.npy')

test_entity = np.load(datadir+'/test_entity.npy')
test_context = np.load(datadir+'/test_context.npy')
test_label = np.load(datadir+'/test_label.npy')
test_fbid = np.load(datadir+'/test_fbid.npy')

linktest = np.load(datadir+'/linktest.npy')



# build model
sess = tf.Session()
model = model.MA("MA")

w2v = util.build_vocab(wv_file, model.word_size)
sess.run(model.initializer)

# model.saver.restore(sess, "parameter/ka/model")

Now = 0
fd = model.fdict(w2v, Now, batch_size, 1, \
    test_entity, test_context, test_label, test_fbid, embedding, False)
fd[model.kprob] = 0.5
sess.run(model.train, feed_dict=fd)
# sess.run(model.train1, feed_dict=fd)
# sess.run(model.train2, feed_dict=fd)

util.test(w2v, model, test_entity, test_context, test_label, test_fbid, embedding, \
    linktest, batch_size, sess, "all")