import copy
import math
import random
import numpy as np
import pyttsx3
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from decimal import Decimal, getcontext


def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


class CallingCounter:

    def __init__(self, func):
        self.func = func
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        return self.func(*args, **kwargs)


def support(sset):
    supp_Set = set()
    a = len(sset)
    for _ in range(a):
        supp_Set = supp_Set.union(sset[_])
    return supp_Set


def S_i(sset, i):
    elements_in_position_i = sset[i]
    return elements_in_position_i


def KSubmodular_template(n, k, m, time, L):
    df = pd.read_pickle('large_dataset.pkl')

    df = df[(df['temperature'] >= 0) & (df['humidity'] >= 0) & (df['light'] >= 0)]

    features = df[['temperature', 'humidity', 'light', 'x_location', 'y_location']]

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=m, random_state=42, init='random', n_init=10)

    cluster_labels = kmeans.fit_predict(features_scaled)

    df['cluster_label'] = cluster_labels

    df_unique = df.drop_duplicates(subset='mote_id', keep='first')

    mote_id_to_label = {mote_id: label for mote_id, label in zip(df_unique['mote_id'], df_unique['cluster_label'])}

    category_groups = defaultdict(list)
    for key, value in mote_id_to_label.items():
        if key in list(range(1, n+1)):
            category_groups[value].append(key)

    category_groups_num = defaultdict(int)
    category_limit_num = defaultdict(int)
    for key, value in category_groups.items():
        category_groups_num[key] = len(value)
        category_limit_num[key] = math.ceil(category_groups_num[key] / L)

    r = np.sum(list(category_limit_num.values()))

    def is_independent_set(vertex_set):
        category_num = {_: 0 for _ in range(m)}
        if not vertex_set:
            return True
        for _ in vertex_set:
            if _ not in mote_id_to_label.keys():
                return False
            category_num[mote_id_to_label[_]] += 1
            if category_num[mote_id_to_label[_]] > category_limit_num[mote_id_to_label[_]]:
                return False
        return True

    vals = {}
    for epoch in df['epoch'].unique():
        if epoch <= time:

            for mote_id in df[df['epoch'] == epoch]['mote_id'].unique():
                if mote_id <= n:
                    sub_df = df[(df['epoch'] == epoch) & (df['mote_id'] == mote_id)]

                    sensor_data = [sub_df['temperature'].iloc[0], sub_df['humidity'].iloc[0],
                                   sub_df['light'].iloc[0]]

                    if epoch in vals.keys():
                        vals[epoch][mote_id] = sensor_data
                    else:
                        vals[epoch] = {mote_id: sensor_data}

    @CallingCounter
    def entropy(kset):
        C = {}
        for t in vals.keys():
            if t < time:
                for p in vals[t].keys():
                    if p < n and p in support(kset):
                        for s, sset in enumerate(kset):
                            if p in sset:
                                x = vals[t][p][s]
                                if x not in C.keys():
                                    C[x] = 0
                                C[x] += 1
        H = 0
        P = []
        for _, __ in C.items():
            pr = 1.0 * __ / (time * k * n)
            P.append(pr)
            H += -pr * math.log(pr)
        return H

    return entropy, r, is_independent_set


def max_monotone_k_sub_Matroid_greedy(value_function, r, is_independent_set):
    s = [set() for _ in range(k)]
    supp_s = set()

    for j in range(r):
        E_s = set()
        print()
        for e in range(1, n+1):
            if e not in supp_s:
                supp_s_copy = copy.copy(supp_s)
                if is_independent_set(supp_s_copy.add(e)):
                    E_s.add(e)
        max_point = None
        max_position = None
        find_max_value = 0
        for e in E_s:
            for i in range(k):
                add_e_to_s = copy.deepcopy(s)
                add_e_to_s[i].add(e)
                f_add_e_to_s = value_function(add_e_to_s)
                if f_add_e_to_s > find_max_value:
                    find_max_value = f_add_e_to_s
                    max_point = e
                    max_position = i
        s[max_position].add(max_point)
        supp_s.add(max_point)
        f_s = value_function(s)
    f_s = value_function(s)
    return f_s, value_function.count


def max_monotone_k_sub_Matroid_Random(value_function, r, is_independent_set):
    X = [set() for _ in range(k)]
    supp_X = set()
    f_X = 0
    w = n * k - 1
    j = 1

    while j <= r:
        p = np.zeros((n, k))
        g = np.zeros((n, k))
        E_X = set()
        for e in range(1, n+1):
            if e not in supp_X:
                supp_X_copy = copy.copy(supp_X)
                supp_X_copy.add(e)
                if is_independent_set(supp_X_copy):
                    E_X.add(e)
        for e in E_X:
            for i in range(k):
                add_e_to_X = copy.deepcopy(X)
                add_e_to_X[i].add(e)
                f_add_e_to_X = value_function(add_e_to_X)
                delta_e_i = f_add_e_to_X - f_X
                g[e-1][i] = delta_e_i

                getcontext().prec = 50

                delta_e_i = Decimal(str(delta_e_i))
                w = Decimal(str(w))
                p[e-1][i] = delta_e_i ** w
        eta = np.sum(p)
        if eta != 0:
            p = p / eta
        else:
            _e = random.choice(list(E_X))
            _i = random.choice(range(k))
            p[_e-1][_i] = 1

        random_num = random.uniform(0, 1)
        while random_num == 0 or random_num == 1:
            random_num = random.uniform(0, 1)
        current_p_sum = 0
        continue_or_not = True
        for e in E_X:
            for i in range(k):
                current_p_sum += p[e-1][i]
                if random_num <= current_p_sum:
                    X[i].add(e)
                    supp_X.add(e)
                    f_X = value_function(X)
                    j += 1
                    continue_or_not = False
                    break
            if not continue_or_not:
                break
    f_X = value_function(X)
    return f_X, value_function.count


def max_monotone_k_sub_Matroid_RRandom(value_function, r, is_independent_set, delta):
    X = [set() for _ in range(k)]
    supp_X = set()
    f_X = 0
    w = n * k - 1
    j = 1

    while j <= r:
        p = np.zeros((n, k))
        g = np.zeros((n, k))
        R_element_num = math.ceil(min(
            (n - j + 1) * (math.log(r / delta)) / (r - j + 1),
            n
        ))
        R = random.sample(list(range(1, n+1)), R_element_num)
        E_X = set()
        for e in R:
            if e not in supp_X:
                supp_X_copy = copy.copy(supp_X)
                supp_X_copy.add(e)
                if is_independent_set(supp_X_copy):
                    E_X.add(e)
        for e in E_X:
            for i in range(k):
                add_e_to_X = copy.deepcopy(X)
                add_e_to_X[i].add(e)
                f_add_e_to_X = value_function(add_e_to_X)
                delta_e_i = f_add_e_to_X - f_X
                g[e-1][i] = delta_e_i

                getcontext().prec = 50

                delta_e_i = Decimal(str(delta_e_i))
                w = Decimal(str(w))
                p[e-1][i] = delta_e_i ** w
        eta = np.sum(p)
        if eta != 0:
            p = p / eta
        else:
            _e = random.choice(list(E_X))
            _i = random.choice(range(k))
            p[_e-1][_i] = 1

        random_num = random.uniform(0, 1)
        while random_num == 0 or random_num == 1:
            random_num = random.uniform(0, 1)
        current_p_sum = 0
        continue_or_not = True
        for e in E_X:
            for i in range(k):
                current_p_sum += p[e-1][i]
                if random_num <= current_p_sum:
                    X[i].add(e)
                    f_X = value_function(X)
                    supp_X.add((e))
                    j += 1
                    continue_or_not = False
                    break
            if not continue_or_not:
                break
        f_X = value_function(X)
    return f_X, value_function.count


if __name__ == '__main__':
    n = 54
    k = 3
    m = 3
    # time = 65535
    time = 500
    L = 5

    (value_function, r, is_independent_set) = KSubmodular_template(n, k, m, time, L)

    a = max_monotone_k_sub_Matroid_Random(value_function, r, is_independent_set)

    b = max_monotone_k_sub_Matroid_RRandom(value_function, r, is_independent_set, 0.1)

    c = max_monotone_k_sub_Matroid_RRandom(value_function, r, is_independent_set, 0.2)

