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


def max_monotone_k_sub_individual_greedy(n, value_function, B_i):
    numerical_record = []

    for __ in range(20):
        k = len(B_i)
        s = [set() for _ in range(k)]
        f_s = 0
        feasible_position = list(range(k))
        B_total = np.sum(B_i)

        t = 0
        while len(feasible_position) != 0 and t != B_total:
            find_max_value = 0
            max_point = None
            max_position = None
            for i in range(n):
                supp_s = support(s)
                if i in supp_s:
                    continue
                else:
                    for j in feasible_position:
                        add_e_to_s = copy.deepcopy(s)
                        add_e_to_s[j].add(i)
                        f_add_e_to_s = value_function(add_e_to_s)
                        if f_add_e_to_s > find_max_value:
                            find_max_value = f_add_e_to_s
                            max_point = i
                            max_position = j
            s[max_position].add(max_point)
            f_s = find_max_value
            if len(s[max_position]) == B_i[max_position]:
                feasible_position.remove(max_position)
            t += 1

        numerical_record.append(f_s)

    return numerical_record, value_function.count / 20


def max_monotone_k_sub_Individual_Random(n, value_function, B_i):
    numerical_record = []

    for __ in range(20):
        k = len(B_i)
        S = [set() for _ in range(k)]
        f_S = 0
        B = np.sum(B_i)
        t = n * k - 1

        for j in range(B):
            p = np.zeros((n, k))
            y = np.zeros((n, k))
            f = np.zeros((n, k))
            feasible_position = []
            for i in range(k):
                if len(S_i(S, i)) <= B_i[i] - 1:
                    for e in range(n):
                        supp_S = support(S)
                        if e not in supp_S:
                            feasible_position.append(i)
                            add_e_to_S = copy.deepcopy(S)
                            add_e_to_S[i].add(e)
                            f_add_e_to_S = value_function(add_e_to_S)
                            delta_e_i = f_add_e_to_S - f_S
                            y[e][i] = delta_e_i
                            p[e][i] = delta_e_i ** t
                            f[e][i] = f_add_e_to_S
            if not feasible_position:
                break
            beta = np.sum(p)
            if beta != 0:
                p = p / beta
            else:
                diff_E_S = list(set(range(n)) - support(S))
                e = random.choice(diff_E_S)
                i = random.choice(feasible_position)
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
    return numerical_record, value_function.count / 20


def max_monotone_k_sub_Individual_RRandom(n, value_function, B_i, delta):
    numerical_record = []

    for __ in range(20):
        k = len(B_i)
        S = [set() for _ in range(k)]
        f_S = 0
        B = np.sum(B_i)
        t = n*k - 1

        for j in range(B):
            p = np.zeros((n, k))
            y = np.zeros((n, k))
            f = np.zeros((n, k))
            feasible_position = []
            for i in range(k):
                if len(S_i(S, i)) <= B_i[i] - 1:
                    feasible_position.append(i)
            if not feasible_position:
                break
            R_elements_num = min(
                max([int((n-len(S_i(S, i)))*math.log(B/delta)/(B_i[i]-len(S_i(S, i)))) for i in feasible_position]),
                n - len(support(S))
            )
            diff_E_S = list(set(range(n)) - support(S))
            if R_elements_num <= len(diff_E_S):
                R = set(random.sample(diff_E_S, R_elements_num))
            else:
                R = diff_E_S
            feasible_position = []
            for i in range(k):
                if len(S_i(S, i)) <= B_i[i] - 1:
                    for e in R:
                        supp_S = support(S)
                        if e not in supp_S:
                            feasible_position.append(i)
                            add_e_to_S = copy.deepcopy(S)
                            add_e_to_S[i].add(e)
                            f_add_e_to_S = value_function(add_e_to_S)
                            delta_e_i = f_add_e_to_S - f_S
                            y[e][i] = delta_e_i
                            p[e][i] = delta_e_i**t
                            f[e][i] = f_add_e_to_S
            if not feasible_position:
                break
            beta = np.sum(p)
            if beta != 0:
                p = p / beta
            else:
                diff_E_S = list(set(range(n))-support(S))
                e = random.choice(diff_E_S)
                i = random.choice(feasible_position)
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
    return numerical_record, value_function.count / 20


if __name__ == '__main__':
    n = 54
    B_i = [1, 1, 1]
    B_plus = B_i
    B_total = sum(B_i)
    k = len(B_i)
    time = 500

    value_function = KSubmodular_template(n, B_i, time)

    a = max_monotone_k_sub_individual_greedy(n, value_function, B_i)

    b = max_monotone_k_sub_Individual_RRandom(n, value_function, B_i, 0.1)

    c = max_monotone_k_sub_Individual_RRandom(n, value_function, B_i, 0.2)
