from typing import List


def hello(name: str) -> str:
    if name == None or name == '':
        return 'Hello!'
    return 'Hello, ' + name + '!'


def int_to_roman(num: int) -> str:
    convert = {1:'I', 5:'V', 10:'X', 50:'L', 100:'C', 500:'D', 1000:'M'}
    nm = str(num)
    ans = ''
    for idx,let in enumerate(nm):
        if let < '5':
            if let == '4':
                ans += (convert[10**(len(nm)-idx-1)] + convert[10**(len(nm)-idx-1)*5])
            elif let == '0':
                continue
            else:
                ans += convert[10**(len(nm)-idx-1)]*int(let)
        else:
            if let == '9':
                ans += (convert[10**(len(nm)-idx-1)] + convert[10**(len(nm)-idx)])
            else:
                ans += (convert[10**(len(nm)-idx-1)*5] + convert[10**(len(nm)-idx-1)]*(int(let)-5))


def longest_common_prefix(strs_input: List[str]) -> str:
    if len(strs_input) == 0:
        return ''
    lst = []
    ans = ''
    for i in range(len(strs_input)):
        lst.append(0)
        for j in strs_input[i]:
            if j == ' ' or j == '\t' or j == '\n':
                lst[i]+=1
            else:
                break
    while(True):
        for i in range(len(lst)-1):
            if lst[i] == len(strs_input[i]) or lst[i+1] == len(strs_input[i+1]):
                return ans
            elif strs_input[i][lst[i]] == strs_input[i+1][lst[i+1]]:
                lst[i]+=1
            else:
                return ans[:len(ans)]
        lst[len(lst)-1]+=1
        ans+=strs_input[0][lst[0]-1]


def primes() -> int:
    simple = 2
    flag = False
    while(True):
        for i in range(2,simple):
            if not simple % i:
                simple += 1 
                flag = True
                break
        if flag:
            flag = False
            continue
        yield simple
        simple += 1


class BankCard:
    def __init__(self, total_sum: int, balance_limit=1000000):
        self.total_sum = total_sum
        self.balance_limit = balance_limit
    def __str__(self):
        return 'To learn the balance call balance.'
    def put(self,sum_put: int):
        self.total_sum += sum_put
        print(f'You put {sum_put} dollars')
        return self
    def __add__(self,x):
        self.total_sum += x.total_sum
        self.balance_limit = max(self.balance_limit,x.balance_limit)
        return self
    def __call__(self,sum_spent):
        if sum_spent <= self.total_sum:
            self.total_sum -= sum_spent
            print(f'You spent {sum_spent} dollars')
            return self
        else:
            print(f'Can\'t spend {sum_spent} dollars')
            raise ValueError
    def __getattr__(self,name):
        return self.name
    def __getattribute__(self, name):
        if name == 'balance':
            self.balance_limit -= 1
            if self.balance_limit == -1:
                print('Balance check limits exceeded.')
                raise ValueError
            return self.total_sum
        return object.__getattribute__(self, name)

