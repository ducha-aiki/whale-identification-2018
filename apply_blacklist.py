with open('data/train.csv', 'r') as fr:
    lines = fr.readlines()
with open('data/blacklist_verified_and_halves.csv', 'r') as bf:
    black_lines = bf.readlines()
    bbb = set()
    for l in black_lines:
        bbb.add(l.strip())
with open('data/train_clean_no_halves.csv', 'w') as ff:
    for l in lines:
        if l.strip().split(',')[0] not in bbb:
            ff.write(l.strip() + '\n')
