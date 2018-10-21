import numpy as np

#using window=15. If not, change all 15 and 30 below as needed

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
            for i in range(min(15, len(a))):
               con[2*i] = a[-i-1]
        else:
            for i in range(min(15, len(a))):
                con[2*i+1] = a[i]
            context.append(con)
        flag = (flag+1) %3

np.save("entity.npy", np.array(entity))
np.save("context.npy", np.array(context))
