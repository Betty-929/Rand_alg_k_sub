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


def S_i(Set, i):
    elements_in_position_i = Set[i]
    return elements_in_position_i


def KSubmodular_template(n, B_i):
    k = len(B_i)

    data = np.zeros((k, n, n))
    for _ in range(k):
        for i in range(n):
            for j in range(n):
                if i < j:
                    if np.random.rand() < 0.1:
                        data[_, i, j] = np.round(np.random.rand(), 2)

    @CallingCounter
    def entropy(init_set):
        _k = len(init_set)

        total_number = []
        for ___ in range(30):
            active_set = [list(s) for s in init_set]
            active_status_set = np.zeros(len(data[0]), dtype=int)
            active_status_set[[item for sublist in active_set for item in sublist]] = 1

            while np.sum(active_status_set) < n:

                if all(not x for x in active_set):
                    break

                for _ in range(_k):

                    neighbors = np.unique(np.where((data[_][active_set[_]] > 0) & (active_status_set == 0))[1])
                    if not len(neighbors):
                        active_set[_] = []
                        continue

                    probabilities = 1 - np.prod(1 - data[0][list(active_set[0]), :][:, list(neighbors)], axis=0)

                    new_active = np.random.rand(len(neighbors)) < probabilities
                    active_status_set[neighbors[new_active]] = 1

                    active_set[_] = list(neighbors[new_active])

            total_number.append(np.sum(active_status_set))

        expect_number = np.mean(total_number)
        return expect_number

    return entropy


def max_monotone_k_sub_Individual_greedy(n, value_function, B_i):
    numerical_record = []

    for __ in range(20):
        _k = len(B_i)
        s = [set() for _ in range(_k)]
        feasible_position = list(range(_k))
        B_total = np.sum(B_i)

        t = 0
        while len(feasible_position) != 0 and t != B_total:
            max_point = None
            max_position = None
            find_max_value = 0
            for e in range(n):
                supp_s = support(s)
                if e in supp_s:
                    continue
                else:
                    for i in feasible_position:
                        add_e_to_s = copy.deepcopy(s)
                        add_e_to_s[i].add(e)
                        f_add_e_to_s = value_function(add_e_to_s)
                        if f_add_e_to_s >= find_max_value:
                            find_max_value = f_add_e_to_s
                            max_point = e
                            max_position = i
            s[max_position].add(max_point)
            t += 1
            if len(s[max_position]) == B_i[max_position]:
                feasible_position.remove(max_position)

        f_s = value_function(s)
        numerical_record.append(f_s)
    evaluation = value_function.count / 20
    return numerical_record, evaluation


def max_monotone_k_sub_Individual_greedyRandom(n, value_function, B_i, delta):
    numerical_record = []

    for __ in range(20):
        _k = len(B_i)
        s = [set() for _ in range(_k)]
        f_s = 0
        feasible_position = list(range(_k))
        B_total = np.sum(B_i)

        for j in range(B_total):
            R = set()
            max_point = None
            max_position = None
            find_max_value = 0
            while 1:
                diff = list(set(range(n)) - support(s) - R)
                new_add = random.choice(diff)
                R.add(new_add)
                for i in feasible_position:
                    add_e_to_s = copy.deepcopy(s)
                    add_e_to_s[i].add(new_add)
                    f_add_e_to_s = value_function(add_e_to_s)
                    if f_add_e_to_s >= find_max_value:
                        find_max_value = f_add_e_to_s
                        max_point = new_add
                        max_position = i
                if len(R) >= min(
                        (n-len(S_i(s, max_position)))*math.log(B_total/delta)/(B_i[max_position] - len(S_i(s, max_position))),
                        n - len(support(s))
                ):
                    s[max_position].add(max_point)
                    f_s = find_max_value
                    if len(s[max_position]) == B_i[max_position]:
                        feasible_position.remove(max_position)
                    break

        numerical_record.append(f_s)
    evaluation = value_function.count / 20
    return numerical_record, evaluation


def max_monotone_k_sub_Individual_Random(n, value_function, B_i):
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
                    for e in range(n):
                        supp_S = support(S)
                        if e not in supp_S:
                            add_e_to_S = copy.deepcopy(S)
                            add_e_to_S[i].add(e)
                            f_add_e_to_S = value_function(add_e_to_S)
                            delta_e_i = f_add_e_to_S - f_S
                            if delta_e_i >= 0:
                                y[e][i] = delta_e_i
                            else:
                                y[e][i] = 0
                            p[e][i] = y[e][i] ** t
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
    evaluation = value_function.count / 20
    return numerical_record, evaluation


def max_monotone_k_sub_Individual_RRandom(n, value_function, B_i, delta):
    numerical_record = []

    for __ in range(20):
        k = len(B_i)
        S = [set() for _ in range(k)]
        f_S = 0
        B = np.sum(B_i)
        t = n*k - 1
        feasible_position = list(range(k))

        for j in range(B):
            p = np.zeros((n, k))
            g = np.zeros((n, k))
            f = np.zeros((n, k))
            R_elements_num = min(
                max([math.ceil((n - len(S_i(S, i))) * math.log(B / delta) / (B_i[i] - len(S_i(S, i)))) for i in
                     feasible_position]),
                n - len(support(S))
            )
            diff_E_S = list(set(range(n)) - support(S))
            R = set(random.sample(diff_E_S, R_elements_num))
            for i in feasible_position:
                for e in R:
                    add_e_to_S = copy.deepcopy(S)
                    add_e_to_S[i].add(e)
                    f_add_e_to_S = value_function(add_e_to_S)
                    delta_e_i = f_add_e_to_S - f_S
                    if delta_e_i >= 0:
                        g[e][i] = delta_e_i
                    else:
                        g[e][i] = 0
                    p[e][i] = g[e][i] ** t
                    f[e][i] = f_add_e_to_S
            beta = np.sum(p)
            if beta != 0:
                p = p / beta
            else:
                e = random.choice(list(R))
                i = random.choice(feasible_position)
                p[e][i] = 1

            random_num = random.uniform(0, 1)
            current_p_sum = 0
            continue_or_not = True
            for e in R:
                for i in feasible_position:
                    current_p_sum += p[e][i]
                    if random_num <= current_p_sum:
                        S[i].add(e)
                        f_S = f[e][i]
                        continue_or_not = False
                        if len(S[i]) == B_i[i]:
                            feasible_position.remove(i)
                        break
                if not continue_or_not:
                    break
        numerical_record.append(f_S)
    evaluation = value_function.count / 20
    return numerical_record, evaluation


if __name__ == "__main__":

    n = 30
    B_i = [1, 1, 1]
    B_plus = B_i
    B_total = np.sum(B_i)

    value_function = KSubmodular_template(n, B_i)

    a = max_monotone_k_sub_Individual_greedy(n, value_function, B_i)

    b = max_monotone_k_sub_Individual_greedyRandom(n, value_function, B_i, delta=0.2)
