with open('val_cls.txt') as f:
    lines = f.readlines()

img_names = [line[:-1].split()[0] for line in lines]

f1 = open('val.txt', 'w')
for i in range(len(img_names)):
    f1.write('/JPEGImages/' + img_names[i] + '.jpg ')
    f1.write('/SegmentationClass/' + img_names[i] + '.png' + '\n')
