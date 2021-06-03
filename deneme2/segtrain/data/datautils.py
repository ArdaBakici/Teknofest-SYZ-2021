import numpy as np
import random
import itertools
from dataflow import ( DataFlow)



# def group_data(data, group_func):
#     keys = []
#     groups = []
#
#     for k, g in itertools.groupby(sorted(data, key=group_func), group_func):
#         klist = list(k)
#         keys.append(klist[0])
#         groups.append((list(g)))
#     return  list(keys), list(groups)
#
#
# def Splitrandom(ratios, seed=None, group_func=None):
#     def f(data):
#         if (group_func is not None):
#             idx, groups = group_data(data, group_func)
#             #print (idx)
#             #print(groups)
#             dict_idx_group = dict(zip(idx, groups))
#
#
#         else:
#             idx = np.arange(len(data))
#
#         if (seed is not None):
#             random.Random(seed).shuffle(idx)
#         else:
#             random.shuffle(idx)
#         N = len(idx)
#
#         splits_idx = []
#         #print(idx)
#         start = 0
#         for i, r in enumerate(ratios):
#             n = int(N * r)
#
#             end = start + n
#             #print(i, n, start, end)
#
#             if (i == len(ratios) - 1):
#                 splits_idx.append(idx[start:])
#             else:
#                 splits_idx.append(idx[start:end])
#             start = end
#
#
#         splits = []
#         for si in splits_idx:
#             asplit = []
#             #print ('S',si)
#             for k in si:
#                 if (group_func is not None):
#                     asplit.extend(dict_idx_group[k])
#                 else:
#                     asplit.append(data[k])
#             splits.append(asplit)
#
#         return splits
#
#     return f



from sklearn.model_selection import GroupShuffleSplit

def Splitrandom (ratios, seed=None, group_func=None):

    def get_group(data):
        if (group_func is not None):
            groups = [group_func(d) for d in data]
        else:
            groups = np.arange(len(data))
        return groups

    def slice(data, idx):
        return [data[id] for id in idx]


    def f(data):

        def group_two(data, ratio_pair):
            groups = get_group(data)
            gss = GroupShuffleSplit(n_splits=1, train_size=ratio_pair[0], test_size=ratio_pair[1], random_state=seed)
            train_idx, test_idx  = next(gss.split(data, groups=groups))
            return slice(data,train_idx) , slice(data,test_idx)

        ratio_new = (ratios[0], np.sum(ratios[1:]))
        train, valtest = group_two(data, ratio_new)
        ratio_new =(ratios[1]/np.sum(ratios[1:]),ratios[2]/np.sum(ratios[1:]))
        val, test = group_two(valtest, ratio_new)
        return train, val, test


    return f


class LabelMap2ProbabilityMap(DataFlow):
    """
    Convert label map to probability map
    """
    def __init__(self, ds, label_map_index, num_classes):
        self.ds = ds
        self.label_map_index = label_map_index
        self.n_class = num_classes

    def size(self):
        return self.ds.size()

    @staticmethod
    def labelmap2probmap_( label_map, n_class):
        s = label_map.shape
        # convert label map to probability map
        pmap = np.zeros((s[0], s[1], n_class))
        for i in range(n_class):
            np.place(pmap[:, :, i], label_map == i, 1)
        return pmap

    def get_data(self):
        for d in self.ds.get_data():
            label_map = d[self.label_map_index]
            s=label_map.shape
            # convert label map to probability map
            pmap = np.zeros((s[0],s[1],self.n_class))
            for i in range(self.n_class):
                np.place(pmap[:, :, i], label_map == i, 1)
            d[self.label_map_index] = pmap
            yield d




def write_text(out_file, text):
    with open(out_file,'w') as f:
        f.write(text)



if( __name__=='__main__'):
    x=['p1_b', 'p1_a', 'p2_c', 'p3_5', 'p2_t','p1_y','p3_9']
    id_f =lambda x: str(x[1])
    Splitter = Splitrandom ((0.6,0.2,0.2),seed=44,group_func=id_f)
    out = Splitter(x)
    print (out)