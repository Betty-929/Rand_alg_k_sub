import copy
import math
import random
import numpy as np


class CallingCounter:
    def __init__(self, func):
        self.func = func
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        return self.func(*args, **kwargs)


def support(Set):
    supp_Set = set()
    a = len(Set)
    for _ in range(a):
        supp_Set = supp_Set.union(Set[_])
    return supp_Set


def S_i(Set, i):
    elements_in_position_i = Set[i]
    return elements_in_position_i


def KSubmodular_template(n, B_i, time):
    k = len(B_i)

    vals = [[[0 for _ in range(n)] for __ in range(time)] for ___ in range(k)]
    with open('data.txt', 'r') as txtfile:
        lines = txtfile.readlines()
        for line in lines:
            row = line.split()
            if len(row) >= 8:
                if int(row[2]) <= time:
                    for s in range(k):
                        if int(row[3]) <= n:
                            vals[s][int(row[2]) - 1][int(row[3]) - 1] = row[s + 4]

    @CallingCounter
    def entropy(KSet):
        C = {}
        t = 0
        while t < time:
            for p in range(n):
                supp_KSet = support(KSet)
                if p in supp_KSet:
                    for s in range(k):
                        a = S_i(KSet, s)
                        if p in a:
                            x = vals[s][t][p]
                            if x not in C.keys():
                                C[x] = 0
                            C[x] += 1
            t += 1
        H = 0
        P = []
        for _, __ in C.items():
            pr = 1.0 * __ / (time * k * n)
            P.append(pr)
            H += -pr * math.log(pr)
        return H

    return entropy


# def max_monotone_k_sub_Total_greedy(n, value_function, B_total, k):
#
#     numerical_record = []
#
#     for __ in range(1):
#         S = [set() for _ in range(k)]
#
#         find_max_value = 0
#         max_point = 0
#         max_position = 0
#         for _ in range(B_total):
#             for i in range(n):
#                 supp_s = support(S)
#                 if i in supp_s:
#                     continue
#                 else:
#                     for j in range(k):
#                         add_e_to_s = copy.deepcopy(S)
#                         add_e_to_s[j].add(i)
#                         f_add_e_to_s = value_function(add_e_to_s)
#                         if f_add_e_to_s > find_max_value:
#                             find_max_value = f_add_e_to_s
#                             max_point = i
#                             max_position = j
#             S[max_position].add(max_point)
#         f_s = find_max_value
#         numerical_record.append(f_s)
#
#     return numerical_record, value_function.count


# def max_monotone_k_sub_Total_greedyRandom(n, value_function, B_total, k, delta):
#     numerical_record = []
#
#     for __ in range(1):
#         S = [set() for _ in range(k)]
#         f_s = 0
#
#         for j in range(B_total):
#             diff = list(set(range(n)) - support(S))
#             R_elements_num = math.ceil(min((n - j + 1) * (math.log(B_total / delta)) / (B_total - j + 1), n - len(support(S))))
#             if R_elements_num > len(diff):
#                 R = diff
#             else:
#                 R = list(set(random.sample(diff, R_elements_num)))
#
#             find_max_value = 0
#             max_point = None
#             max_position = None
#             for e in R:
#                 for i in range(k):
#                     add_e_to_s = copy.deepcopy(S)
#                     add_e_to_s[i].add(e)
#                     f_add_e_to_s = value_function(add_e_to_s)
#                     if f_add_e_to_s > find_max_value:
#                         find_max_value = f_add_e_to_s
#                         max_point = e
#                         max_position = i
#             S[max_position].add(max_point)
#             f_s = find_max_value
#
#         f_s = value_function(S)
#         numerical_record.append(f_s)
#     evaluation = value_function.count
#     return numerical_record, evaluation


def max_monotone_k_sub_Total_Random(n, value_function, B_total, k):
    numerical_record = []

    for __ in range(1):
        S = [set() for _ in range(k)]
        f_S = 0
        t = n*k - 1

        for j in range(B_total):
            p = np.zeros((n, k))
            y = np.zeros((n, k))
            f = np.zeros((n, k))
            for e in range(n):
                supp_S = support(S)
                if e not in supp_S:
                    for i in range(k):
                        add_e_to_S = copy.deepcopy(S)
                        add_e_to_S[i].add(e)
                        f_add_e_to_S = value_function(add_e_to_S)
                        delta_e_i = f_add_e_to_S - f_S
                        y[e][i] = delta_e_i
                        p[e][i] = delta_e_i**t
                        f[e][i] = f_add_e_to_S
            beta = np.sum(p)
            if beta != 0:
                p = p / beta
            else:
                diff_E_S = list(set(range(n))-support(S))
                e = random.choice(diff_E_S)
                i = random.choice(range(k))
                p[e][i] = 1

            random_num = random.uniform(0, 1)
            current_p_sum = 0
            continue_or_not = True
            for e in range(n):
                for i in range(k):
                    current_p_sum += p[e][i]
                    if random_num <= current_p_sum:
                        S[i].add(e)
                        f_S = f[e][i]
                        continue_or_not = False
                        break
                if not continue_or_not:
                    break
        numerical_record.append(f_S)
    return numerical_record, value_function.count


def max_monotone_k_sub_Total_RRandom(n, value_function, B_total, k, delta):
    numerical_record = []

    for __ in range(1):
        S = [set() for _ in range(k)]
        f_S = 0
        t = n*k - 1

        for j in range(B_total):
            p = np.zeros((n, k))
            y = np.zeros((n, k))
            f = np.zeros((n, k))
            R_elements_num = min(int((n-j+2)*math.log(B_total/delta)/(B_total-j+2)), n)
            diff_E_S = list(set(range(n)) - support(S))
            if R_elements_num <= len(diff_E_S):
                R = set(random.sample(diff_E_S, R_elements_num))
            else:
                R = diff_E_S
            for e in R:
                supp_S = support(S)
                if e not in supp_S:
                    for i in range(k):
                        add_e_to_S = copy.deepcopy(S)
                        add_e_to_S[i].add(e)
                        f_add_e_to_S = value_function(add_e_to_S)
                        delta_e_i = f_add_e_to_S - f_S
                        y[e][i] = delta_e_i
                        p[e][i] = delta_e_i**t
                        f[e][i] = f_add_e_to_S
            beta = np.sum(p)
            if beta != 0:
                p = p / beta
            else:
                supp_S = support(S)
                diff_E_S = list(set(range(n))-supp_S)
                e = random.choice(diff_E_S)
                i = random.choice(range(k))
                p[e][i] = 1

            random_num = random.uniform(0, 1)
            current_p_sum = 0
            continue_or_not = True
            for e in range(n):
                for i in range(k):
                    current_p_sum += p[e][i]
                    if random_num <= current_p_sum:
                        S[i].add(e)
                        f_S = f[e][i]
                        continue_or_not = False
                        break
                if not continue_or_not:
                    break
        numerical_record.append(f_S)
    return numerical_record, value_function.count


if __name__ == '__main__':
    n = 54
    B_i = [1, 1, 1]
    B_plus = B_i
    B_total = sum(B_i)
    k = len(B_i)
    time = 500

    value_function = KSubmodular_template(n, B_i, time)

    a = max_monotone_k_sub_Total_Random(n, value_function, B_total, k)

    b = max_monotone_k_sub_Total_RRandom(n, value_function, B_total, k, 0.1)

    c = max_monotone_k_sub_Total_RRandom(n, value_function, B_total, k, 0.2)

