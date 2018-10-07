import sys
import numpy as np
import argparse
import model
import util
import tensorflow as tf



####### parse arguments
parser = argparse.ArgumentParser()

group = parser.add_mutually_exclusive_group()
group.add_argument("--train", action="store_true")
group.add_argument("--test", action="store_true")
group.add_argument("--direct", action="store_true")
parser.add_argument("--model", action="store", required=True,
    help="which model to use")
parser.add_argument("--data_dir", action="store", required=True,
    help="the directory to data files")
parser.add_argument("--direct_dir", action="store", required=False,
    help="the directory to direct output files")
parser.add_argument("--w2v_file", action="store", required=True,
    help="the path to the word vector file")
parser.add_argument("--save_dir", action="store", required=True,
    help="the directory to save trained models")
parser.add_argument("--load_model", action="store",
    help="the path to the model parameter files to be loaded")

args = parser.parse_args()
training = args.train
direct = args.direct
modelname = args.model
datadir = args.data_dir
directdir = args.direct_dir
w2vfile = args.w2v_file
savedir = args.save_dir

batch_size = 1000
batch_manual = 100
iter_num = 10000
check_freq = 1000



####### load data
util.printlog("Loading data")
embedding = np.load(datadir+'/embedding.npy')

if training:

    train_entity = np.load(datadir+'/train_entity.npy')
    train_context = np.load(datadir+'/train_context.npy')
    train_label = np.load(datadir+'/train_label.npy')
    train_fbid = np.load(datadir+'/train_fbid.npy')

    valid_entity = np.load(datadir+'/valid_entity.npy')
    valid_context = np.load(datadir+'/valid_context.npy')
    valid_label = np.load(datadir+'/valid_label.npy')
    valid_fbid = np.load(datadir+'/valid_fbid.npy')

    linkvalid = np.load(datadir+'/linkvalid.npy')

    train_size = len(train_entity)

elif direct:

    direct_entity = np.load(directdir+'/entity.npy')
    direct_context = np.load(directdir+'/context.npy')


else:

    test_entity = np.load(datadir+'/test_entity.npy')
    test_context = np.load(datadir+'/test_context.npy')
    test_label = np.load(datadir+'/test_label.npy')
    test_fbid = np.load(datadir+'/test_fbid.npy')

    manual_entity = np.load(datadir+'/manual_entity.npy')
    manual_context = np.load(datadir+'/manual_context.npy')
    manual_label = np.load(datadir+'/manual_label.npy')
    manual_fbid = np.load(datadir+'/manual_fbid.npy')

    linktest = np.load(datadir+'/linktest.npy')
    linkmanual = np.load(datadir+'/linkmanual.npy')



####### build model
if modelname=="SA":
    model = model.SA("SA")
elif modelname=="MA":
    model = model.MA("MA")
elif modelname=="KA":
    model = model.KA("KA")
elif modelname=="KA+D":
    model = model.KA_D("KA+D")
else:
    raise ValueError("No such model!")

sess = tf.Session()
w2v = util.build_vocab(w2vfile, model.word_size)
sess.run(model.initializer)

if args.load_model:
    model.saver.restore(sess, args.load_model)
elif not training:
    raise ValueError("Must load a model for testing!")


####### direct
if direct:
    util.printlog("Begin computing direct outputs")
    util.direct(w2v, sess, model, direct_entity, direct_context, embedding)

####### train
elif training:
    util.printlog("Begin training")

    for i in range(iter_num):

        if i%check_freq==0:
            util.printlog("Validating after running "+str(i*batch_size/train_size)+" epoches")
            util.test(w2v, model, valid_entity, valid_context, valid_label, valid_fbid, \
                embedding, linkvalid, batch_size, sess, "all")
            model.saver.save(sess, savedir+"/model"+str(i))

        fd = model.fdict(w2v, (i*batch_size)%train_size, batch_size, 1, \
            train_entity, train_context, train_label, train_fbid, embedding, False)
        fd[model.kprob] = 0.5
        sess.run(model.train, feed_dict=fd)

        if i%(train_size/batch_size/10)==0:
            util.printlog("Epoch %d, Batch %d" \
                %((i*batch_size)/train_size, (i*batch_size)%train_size/batch_size))





####### test
else:
    util.printlog("Test on the wiki-auto test set")
    util.test(w2v, model, test_entity, test_context, test_label, test_fbid, \
        embedding, linktest, batch_size, sess, "all")
    util.test(w2v, model, test_entity, test_context, test_label, test_fbid, \
        embedding, linktest, batch_size, sess, "succ")
    util.test(w2v, model, test_entity, test_context, test_label, test_fbid, \
        embedding, linktest, batch_size, sess, "miss")
    util.test(w2v, model, test_entity, test_context, test_label, test_fbid, \
        embedding, linktest, batch_size, sess, "person")
    util.test(w2v, model, test_entity, test_context, test_label, test_fbid, \
        embedding, linktest, batch_size, sess, "organization")
    util.test(w2v, model, test_entity, test_context, test_label, test_fbid, \
        embedding, linktest, batch_size, sess, "location")

    util.printlog("Test on the wiki-man test set")
    util.test(w2v, model, manual_entity, manual_context, manual_label, manual_fbid, \
        embedding, linkmanual, batch_manual, sess, "all")
    util.test(w2v, model, manual_entity, manual_context, manual_label, manual_fbid, \
        embedding, linkmanual, batch_manual, sess, "succ")
    util.test(w2v, model, manual_entity, manual_context, manual_label, manual_fbid, \
        embedding, linkmanual, batch_manual, sess, "miss")
    util.test(w2v, model, manual_entity, manual_context, manual_label, manual_fbid, \
        embedding, linkmanual, batch_manual, sess, "person")
    util.test(w2v, model, manual_entity, manual_context, manual_label, manual_fbid, \
        embedding, linkmanual, batch_manual, sess, "organization")
    util.test(w2v, model, manual_entity, manual_context, manual_label, manual_fbid, \
        embedding, linkmanual, batch_manual, sess, "location")