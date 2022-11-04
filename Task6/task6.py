def check(x: str, file: str):
    words = x.lower().split(' ')
    words_unique = set(x.lower().split(' '))
    dct = dict()
    with open(file,'w') as fin:
        for word in words:
            if word in words_unique:
                if dct.get(word,-1) == -1:
                    dct[word] = 1
                else:
                    dct[word] += 1
        dct = dict(sorted(dct.items()))
        for key,item in dct.items():
            print(key,item,sep = ' ',file = fin)

