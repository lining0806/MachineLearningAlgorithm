# -*- coding: utf-8 -*-
'''
HMM（隐马尔可夫模型）是用来描述隐含未知参数的统计模型
举一个经典的例子：
一个东京的朋友每天根据天气{下雨,天晴}决定当天的活动{公园散步,购物,清理房间}中的一种
我每天只能在twitter上看到她发的推“啊，我前天公园散步、昨天购物、今天清理房间了！”
那么我可以根据她发的twitter推断东京这三天的天气
在这个例子里，显状态是活动，隐状态是天气
求解最可能的隐状态序列是HMM的三个典型问题之一，通常用Viterbi算法解决
Viterbi算法就是求解HMM上的最短路径（-log(prob)，也即是最大概率）的算法
'''

# HMM描述 lambda = (states, observations, start_probability, transition_probability, emission_probability)
states = ('Rainy', 'Sunny')

observations = ('walk', 'shop', 'clean')

start_probability = {'Rainy': 0.6, 'Sunny': 0.4}

transition_probability = {
    'Rainy' : {'Rainy': 0.7, 'Sunny': 0.3},
    'Sunny' : {'Rainy': 0.4, 'Sunny': 0.6},
    }

emission_probability = {
    'Rainy' : {'walk': 0.1, 'shop': 0.4, 'clean': 0.5},
    'Sunny' : {'walk': 0.6, 'shop': 0.3, 'clean': 0.1},
}

# 打印路径概率表
def print_dptable(V):
    print '',
    for t in range(len(V)):
        print "%7d" % t,
    print ''
    for y in V[0].keys():
        print "%.5s:" % y,
        for t in range(len(V)):
            print "%.7s" % ("%f" % V[t][y]),
        print ''

def viterbi(stas, obs, start_p, trans_p, emit_p):
    '''
    :param stas:隐状态
    :param obs:观测序列
    :param start_p:初始概率（隐状态）
    :param trans_p:转移概率（隐状态）
    :param emit_p:发射概率（隐状态表现为显状态的概率）
    :return:
    思路：
    定义V[时间][今天天气] = 概率，注意今天天气指的是，前几天的天气都确定下来了（概率最大）今天天气是X的概率，这里的概率就是一个累乘的概率了
    因为第一天我的朋友去散步了，所以第一天下雨的概率V[第一天][下雨] = 初始概率[下雨] * 发射概率[下雨][散步] = 0.6 * 0.1 = 0.06，同理可得V[第一天][天晴] = 0.24。从直觉上来看，因为第一天朋友出门了，她一般喜欢在天晴的时候散步，所以第一天天晴的概率比较大，数字与直觉统一了。
    从第二天开始，对于每种天气Y，都有前一天天气是X的概率 * X转移到Y的概率 * Y天气下朋友进行这天这种活动的概率。因为前一天天气X有两种可能，所以Y的概率有两个，选取其中较大一个作为V[第二天][天气Y]的概率，同时将今天的天气加入到结果序列中
    比较V[最后一天][下雨]和[最后一天][天晴]的概率，找出较大的哪一个对应的序列，就是最终结果
    '''

    # 路径概率表 V[时间][隐状态] = 概率
    V = [{}]
    # 一个中间变量，代表当前状态是哪个隐状态
    path = {}

    # 初始化初始状态 (对t == 0)
    for y in stas:
        V[0][y] = start_p[y] * emit_p[y][obs[0]]
        path[y] = [y] # 记录初始路径，前面的key对应y状态
    print V
    print path

    # 跑一遍维特比算法 (对 t > 0)
    for t in range(1, len(obs)):
        V.append({})

        new_path = {}
        for y in stas:
            '''隐状态概率 = 前状态是y0的概率 * y0转移到y的概率 * y表现为当前状态的概率'''
            # y的最大概率及对应的前状态sta
            (prob, sta) = max([(V[t - 1][y0] * trans_p[y0][y] * emit_p[y][obs[t]], y0) for y0 in stas])
            # 记录最大隐状态概率
            V[t][y] = prob
            # 记录路径
            new_path[y] = path[sta] + [y] # 记录当前路径，前面的key对应y状态
        print V
        print new_path

        # 不需要保留旧路径
        path = new_path

    print_dptable(V)

    # 找出概率最大的最后状态
    (prob, sta) = max([(V[len(obs) - 1][y], y) for y in stas])
    return prob, path[sta]

def example():
    return viterbi(states,
                   observations,
                   start_probability,
                   transition_probability,
                   emission_probability)

print example()