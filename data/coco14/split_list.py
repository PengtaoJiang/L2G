with open('train_cls.txt') as f:
    lines = f.readlines()

L = len(lines) // 20000
print(L)
for i in range(L):
    f1 = open('train_cls_part{}.txt'.format(i+1), 'w') 
    for j in range(20000):
        line = lines[20000*i + j]
        f1.write(line)
    f1.close()

f1 = open('train_cls_part{}.txt'.format(L+1), 'w') 
for j in range(len(lines)-L*20000):
    line = lines[20000*L + j]
    f1.write(line)
f1.close()
