import math
import numpy as np
import random
import copy


class CallingCounter:
    def __init__(self, func):
        self.func = func
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        return self.func(*args, **kwargs)


def support(k_set):
    supp_k_set = set()
    for _ in range(len(k_set)):
        supp_k_set = supp_k_set.union(k_set[_])
    return supp_k_set


def KSubmodular_template(n, B_i):
    k = len(B_i)

    data = np.zeros((k, n, n))
    for _ in range(k):
        for i in range(n):
            for j in range(n):
                if i < j:
                    if np.random.rand() < 0.2:
                        data[_, i, j] = np.round(np.random.rand(), 2)

    @CallingCounter
    def entropy(init_set):
        _k = len(init_set)

        total_number = []
        for _ in range(30):
            active_set = [list(s) for s in init_set]
            active_status_set = np.zeros(len(data[0]), dtype=int)
            active_status_set[[item for sublist in active_set for item in sublist]] = 1

            neighbors_status = [True for __ in range(_k)]
            while np.sum(active_status_set) < len(data[0]):
                if all(not x for x in neighbors_status):
                    break

                for _ in range(_k):

                    neighbors = np.unique(np.where((data[_][active_set[_]] > 0) & (active_status_set == 0))[1])
                    if not len(neighbors):
                        neighbors_status[_] = False
                        continue

                    probabilities = 1 - np.prod(1 - data[0][list(active_set[0]), :][:, list(neighbors)], axis=0)

                    new_active = np.random.rand(len(neighbors)) < probabilities
                    active_status_set[neighbors[new_active]] = 1

                    active_set[_] = neighbors[new_active]

            total_number.append(np.sum(active_status_set))

        expect_number = np.mean(total_number)
        return expect_number

    return entropy


def max_monotone_k_sub_Total_greedy(n, value_function, B_i):
    numerical_record = []

    for __ in range(20):
        _k = len(B_i)
        S = [set() for _ in range(_k)]
        B_total = np.sum(B_i)

        for _ in range(B_total):
            max_point = None
            max_position = None
            find_max_value = 0
            for e in range(n):
                if e in support(S):
                    continue
                else:
                    for i in range(_k):
                        add_e_to_s = copy.deepcopy(S)
                        add_e_to_s[i].add(e)
                        f_add_e_to_s = value_function(add_e_to_s)
                        if f_add_e_to_s >= find_max_value:
                            find_max_value = f_add_e_to_s
                            max_point = e
                            max_position = i
            S[max_position].add(max_point)

        f_s = value_function(S)
        numerical_record.append(f_s)
    f_s = np.mean(numerical_record)
    max_f_s = np.max(numerical_record)
    evaluation = value_function.count / 20
    return f_s, max_f_s, evaluation


def max_monotone_k_sub_Total_greedyRandom(n, value_function, B_i, delta):
    numerical_record = []

    for __ in range(20):
        _k = len(B_i)
        S = [set() for _ in range(_k)]
        f_s = 0
        B_total = np.sum(B_i)

        for j in range(B_total):
            diff = list(set(range(n)) - support(S))
            R_elements_num = math.ceil(min((n - j + 1) * (math.log(B_total / delta)) / (B_total - j + 1), n - len(support(S))))
            if R_elements_num > len(diff):
                R = diff
            else:
                R = list(set(random.sample(diff, R_elements_num)))

            find_max_value = 0
            max_point = None
            max_position = None
            for e in R:
                for i in range(_k):
                    add_e_to_s = copy.deepcopy(S)
                    add_e_to_s[i].add(e)
                    f_add_e_to_s = value_function(add_e_to_s)
                    if f_add_e_to_s > find_max_value:
                        find_max_value = f_add_e_to_s
                        max_point = e
                        max_position = i
            S[max_position].add(max_point)
            f_s = find_max_value

        f_s = value_function(S)
        numerical_record.append(f_s)
    f_s = np.mean(numerical_record)
    max_f_s = np.max(numerical_record)
    evaluation = value_function.count / 20
    return f_s, max_f_s, evaluation


def max_monotone_k_sub_Total_Random(n, value_function, B_i):

    numerical_record = []

    for __ in range(20):
        k = len(B_i)
        S = [set() for _ in range(k)]
        f_S = value_function(S)
        B = np.sum(B_i)
        t = n*k - 1

        for j in range(B):
            p = np.zeros((n, k))
            y = np.zeros((n, k))
            for e in range(n):
                if e not in support(S):
                    for i in range(k):
                        add_e_to_S = copy.deepcopy(S)
                        add_e_to_S[i].add(e)
                        f_add_e_to_S = value_function(add_e_to_S)
                        delta_e_i = f_add_e_to_S - f_S
                        if delta_e_i >= 0:
                            y[e][i] = delta_e_i
                        else:
                            y[e][i] = 0
                        p[e][i] = y[e][i] ** t
            beta = np.sum([np.sum(p[_]) for _ in range(n)])
            if beta != 0:
                for e in range(n):
                    for i in range(k):
                        p[e][i] = p[e][i] / beta
            else:
                diff_E_S = list(set(range(n))-support(S))
                e = random.choice(diff_E_S)
                i = random.choice(range(k))
                p[e][i] = 1

            random_num = random.uniform(0, 1)
            while random_num == 0 or random_num == 1:
                random_num = random.uniform(0, 1)
            current_p_sum = 0
            continue_or_not = True
            for e in range(n):
                for i in range(k):
                    current_p_sum += p[e][i]
                    if random_num <= current_p_sum:
                        S[i].add(e)
                        continue_or_not = False
                        break
                if not continue_or_not:
                    break
        f_S = value_function(S)
        numerical_record.append(f_S)
    f_S = np.mean(numerical_record)
    max_f_s = np.max(numerical_record)
    evaluation = value_function.count / 20
    return f_S, max_f_s, evaluation


def max_monotone_k_sub_Total_RRandom(n, value_function, B_i, delta):
    numerical_record = []

    for __ in range(20):
        k = len(B_i)
        S = [set() for _ in range(k)]
        f_S = value_function(S)
        B = np.sum(B_i)
        t = n*k - 1

        for j in range(B):
            p = np.zeros((n, k))
            y = np.zeros((n, k))
            R_elements_num = min(int((n-j+2)*math.log(B/delta)/(B-j+2)), n)
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
                        if delta_e_i >= 0:
                            y[e][i] = delta_e_i
                        else:
                            y[e][i] = 0
                        p[e][i] = y[e][i]**t
            beta = np.sum(p)
            if beta != 0:
                for e in R:
                    for i in range(k):
                        p[e][i] = p[e][i] / beta
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
                        continue_or_not = False
                        break
                if not continue_or_not:
                    break
        f_S = value_function(S)
        numerical_record.append(f_S)
    f_S = np.mean(numerical_record)
    max_f_s = np.max(numerical_record)
    evaluation = value_function.count / 20
    return f_S, max_f_s, evaluation


if __name__ == "__main__":

    n = 30
    B_i = [1, 1, 1]
    B_plus = B_i
    B_total = np.sum(B_i)

    value_function = KSubmodular_template(n, B_i)

    a = max_monotone_k_sub_Total_greedy(n, value_function, B_i)

    b = max_monotone_k_sub_Total_greedyRandom(n, value_function, B_i, delta=0.2)

    d = max_monotone_k_sub_Total_Random(n, value_function, B_i)

    e = max_monotone_k_sub_Total_RRandom(n, value_function, B_i, 0.2)

