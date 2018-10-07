import numpy as np

entity = []
context = []
with open("raw", 'r') as f:
    flag = 0
    con = ["unk" for _ in range(30)]
    for line in f:
        a = line.split()
        if flag==0:
            entity.append(a)
            con = ["unk" for _ in range(30)]
        elif flag==1:
            for i in range(len(a)):
               con[2*i] = a[-i]
        else:
            for i in range(len(a)):
                con[2*i+1] = a[i]
            context.append(con)
        flag = (flag+1) %3

np.save("entity.npy", np.array(entity))
np.save("context.npy", np.array(context))
