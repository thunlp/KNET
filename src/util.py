import numpy as np


def guess(y, y_, type_size, sess, fd, ctb, th=0.5):
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
            for j in range(type_size):
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
    table = np.transpose(np.tile(ctb, [type_size, 1]))
    true_pos = float(np.sum(np.logical_and(table, np.logical_and(h, showy_))))
    false_pos = float(np.sum(np.logical_and(table, h)))
    true_neg = float(np.sum(np.logical_and(table, showy_)))
    
    effect = float(np.sum(ctb))
    return (float(strict), lma_p, lma_r, true_pos, false_pos, true_neg, effect)


def test(w2v, model, _entity, _context, _label, _fbid, _embedding, \
    _link, batch_size, sess, version):
    true_pos = 0
    false_pos = 0
    true_neg = 0    
    strict = 0
    lma_p = 0
    lma_r = 0
    effect = 0
    #organization 0 person 13 location 54

    full_size = len(_label)
    for i in range(int(full_size/batch_size)):        

        if version=='all':
            table = np.ones([batch_size])
        elif version=='succ':
            table = _link[i*batch_size : (i+1)*batch_size]
        elif version=='miss':
            table = np.ones([batch_size]) - _link[i*batch_size : (i+1)*batch_size]       
        else:
            table = np.zeros([batch_size])
        
        fdt = model.fdict(w2v, i*batch_size, batch_size, 1, \
            _entity, _context, _label, _fbid, _embedding, True)
        t = -1
        if version=='person':
            t = 13
        elif version=='organization':
            t = 0
        elif version=='location':
            t = 54
        if t>-1:            
            for j in range(batch_size):         
                if _label[i*batch_size+j][t]==1:
                    table[j] = 1
                    
        fdt[model.kprob] = 1.0
        
        result = guess(model.t, model.t_, model.type_size, sess, fdt, table)
        
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
    print('loose-macro: %f %f %f' \
        %(lma_p/effect, lma_r/effect, (2*lma_p*lma_r)/(lma_p+lma_r)/effect))
    print('loose-micro: %f %f %f\n\n' \
        %(precision, recall, (precision*recall*2)/(precision+recall)))


def build_vocab(wvfile, word_size):
    w2v = {}
    vocab = 0 #size of vocabulary

    with open(wvfile, 'r') as f:
        while True:
            s = f.readline()
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
    return w2v


def dic(w2v, s):
    if s in w2v:
        return w2v[s]
    else:
        return w2v['unk']


def build_disamb(disamb_file):
    disamb = {}
    with open(disamb_file, 'r') as f:
        head = ''
        for ss in f:
            s = ss[:-1]
            if s[0]!='\t':
                head = s
                if s not in disamb:
                    disamb[s] = []
            else:
                disamb[head].append(int(s[1:]))
    return disamb