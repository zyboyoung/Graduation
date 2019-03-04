import os
import time
from functools import wraps


# 计算程序运行时间的装饰器
def cal_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print('-' * 8)
        print('start time: ', time.asctime(time.localtime(start)))
        print('end time:   ', time.asctime(time.localtime(end)))
        print('-' * 8)
        cost_time = end - start
        if cost_time < 1:
            print(func.__name__, '{:.5f}'.format(cost_time * 1000), 'ms')
        else:
            print(func.__name__, str(cost_time // 3600), 'h', str(cost_time % 3600 // 60), 'min',
                  '{:.2f}'.format(cost_time % 60), 's')
        print('-' * 8)
        return result
    return wrapper


@cal_time
def gen_structure(data_dir='../GraphProt_CLIP_sequences/', target_dir='../GraphProt_CLIP_ss/',
                  pwd=os.path.abspath('..') + '/GraphProt_CLIP_sequences/'):
    """
    调用RNAfold的终端接口，生成包含了所有预测的RNA二级结构信息的文件，此时保留了RNA所有长度的序列
    :return: None
    """
    file_list = os.listdir(data_dir)
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    for file in file_list:
        ss_file = target_dir + file
        if os.path.exists(ss_file):
            continue
        else:
            with open(ss_file, 'w') as f:
                command = 'cd ' + pwd + ';RNAfold --noPS <' + file
                print(command)
                result = os.popen(command)
                f.write(result.read())


def padding_sequence(seq, max_len=501, key='N'):
    """
    对于长度超过501bp的序列，只选择其前501bp，对于不足501bp的序列，使用'N'在原序列后填充
    :param seq:
    :param max_len:
    :param key:
    :return:
    """
    seq_len = len(seq)
    if seq_len < max_len:
        gap_len = max_len - seq_len
        new_seq = seq + key * gap_len
        new_seq = ''.join(i for i in new_seq)
    else:
        new_seq = seq[:max_len]
    return new_seq


@cal_time
def split_sequences(source_path='../GraphProt_CLIP_sequences/', target_path='/GraphProt_CLIP_sequences_501bp/'):
    """
    将所有序列统一为一致的长度，501bp，该长度是数据集中所有序列的最大长度
    :param source_path: 源文件所处路径
    :param target_path: 写文件的目标路径，需要先创建该文件夹
    :return:
    """
    source_file_list = os.listdir(source_path)
    pwd = os.path.abspath('..') + target_path
    if not os.path.exists(pwd):
        os.mkdir(pwd)
    for file in source_file_list:
        target_file = pwd + file
        if os.path.exists(target_file):
            continue
        else:
            with open(source_path + file, 'r') as source, open(target_file, 'w') as target:
                for line in source:
                    if line[0] == '>':
                        target.write(line)
                    else:
                        line = line.split()[0]
                        new_line = padding_sequence(seq=line, max_len=501, key='N')
                        target.write(new_line)
                        target.write('\n')
                print(target_file + ' completed')


if __name__ == '__main__':
    # split_sequences()

    gen_structure(data_dir='../GraphProt_CLIP_sequences_501bp/', target_dir='../GraphProt_CLIP_ss_501bp/',
                  pwd=os.path.abspath('..') + '/GraphProt_CLIP_sequences_501bp/')
