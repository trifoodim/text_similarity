import string
from functools import lru_cache
from timeit import timeit
import argparse
import os
import re
import numpy as np
import time


def test_lev_dist(f: callable, a, b, n=1):
    tm = timeit("f(a, b)", globals={
        'f': f, 'a': a, 'b': b
    }, number=n)
    r = f(a, b)
    print(f'a = {a!r} and b = {b!r} and {f.__name__} = {r} and time = {tm:.4f}')


def my_dist_cached(a, b):
    @lru_cache(maxsize=len(a) * len(b))
    def recursive(i, j):
        if i == 0 or j == 0:
            return max(i, j)
        elif a[i - 1] == b[j - 1]:
            return recursive(i - 1, j - 1)
        else:
            return 1 + min(
                recursive(i, j - 1),
                recursive(i - 1, j),
                recursive(i - 1, j - 1)
            )

    return recursive(len(a), len(b))


def arg_parce_fnc():
    """
    Функция распаковки аргументов из CLI
    """
    parser = argparse.ArgumentParser(description='test init')
    parser.add_argument('input_file', type=str, help='Input file name for check')
    parser.add_argument('output_file', type=str, help='Output file name for result')
    args = parser.parse_args()

    return args.input_file, args.output_file


def remove_punctuations(txt, punct=string.punctuation):
    """
    Функция удаления знаков пунктуации
    """
    return ''.join([c for c in txt if c not in punct])


def clean_text(txt):
    """
    Эта функция очистит передаваемый текст, удалив определенные символы перевода строки
    такие как '\n', '\r' и '\'
    """
    txt = txt.replace('\n', ' ').replace('\r', ' ').replace('\'', '')
    txt = remove_punctuations(txt)
    return txt.lower()


def d_lev(s1, s2, c=[], dtype=np.uint32):
    """
    Функция вычисления расстояния Левенштейна между текстами
    """
    e_d = np.arange(len(s2) + 0, dtype=dtype)

    if s1 == s2:
        return 0

    if c is d_lev.__defaults__[0]:
        c = np.ones(3, dtype)

    for i in np.arange(1, len(s1)):
        e_i = np.concatenate(([i], np.zeros(len(s2) - 1, dtype)), axis=0)

        for j in np.arange(1, len(s2) + 0):
            r_cost = 0 if s1[i - 1] == s2[j - 1] else 1

            e_i[j] = np.min([(e_d[j] + 1) * c[0],  # s1[i] - deleted from s1, and inserted to s2
                             (e_i[j - 1] + 1) * c[1],  # s2[j] - inserted to s1, and deleted from s2
                             (e_d[j - 1] + r_cost) * c[2]])  # s1[i] - replaced by s2[j]

        e_d = np.array(e_i, copy=True)

    return e_d[len(e_d) - 1]


def hamming_dist(s1, s2):
    """
    Функция вычисления расстояния Хэмминга
    """
    if s1 == s2:
        return 0

    s1 = np.asarray(list(s1))
    s2 = np.asarray(list(s2))

    l_diff = abs(np.size(s1) - np.size(s2))
    l_min = np.min((np.size(s1), np.size(s2)))

    return np.count_nonzero(s1[:l_min] != s2[:l_min]) + l_diff


def lev_score(s1, s2, dl):
    """
    Функция нормирования расстояния Левенштейна через расстояние Хэмминга
    """
    if s1 == s2:
        return 1.0

    dh = hamming_dist(s1, s2)
    h_norm = dh / np.max((len(s1), len(s2)))

    return 1.0 - dl * h_norm / dh


def survey_metrics(s1, s2):
    dists = {'dl': d_lev(s1, s2), }

    score = {'dl': lev_score(s1, s2, dists['dl']), }

    return np.array(
        [{'name': name, 'd': dists[d], 's': score[s]} for [name, d, s] in zip(mt_names, dists.keys(), score.keys())],
        dtype=object)


def survey(s1, s2):
    output = ""
    results = survey_metrics(s1, s2)

    valid = is_dl_valid(s1, s2, results[0]['d'], results[1]['d'])

    output += "strings : [ \"%s\" -> \"%s\" ]\n\n" % (s1, s2)

    output += "distance: [ " + " | ".join(["%s : %4.2f" % (mt['name'], mt['d']) for mt in results]) + " ]\n"

    output += "similarity: [ " + " | ".join(
        ["%s : %4.2f%%" % (mt['name'], np.multiply(100, mt['s'])) for mt in results])

    output += "verification : [ %s ]\n\n" % valid
    return output


def demo(filename):
    output = "Filename%s: %s\n\n" % (" " * 5, os.path.abspath(filename))

    with open(filename, 'r') as f:
        strings = f.readline().split('\n')

        for i in np.arange(len(strings) - 1):
            if len(strings[i]) > 1 and len(strings[i + 1]) > 1:
                print(survey(strings[i], strings[i + 1]))
                time.sleep(1)

    return output


mt_names = ["Levenshtein Distance"]

if __name__ == '__main__':
    inp_file_name, out_file_name = arg_parce_fnc()

    res = demo(inp_file_name)

    with open(out_file_name, 'w') as file1:
        file1.write(res)
