import numpy as np
import lap
import os

def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b

def compute_dis(position1_in, position2_in):
    position1 = position1_in[1:4]
    position2 = position2_in[1:4]
    position1 = np.array(position1)
    position2 = np.array(position2)
    return np.sqrt(np.dot(position1-position2, position1-position2))

def compute_cost_martix_position(positions_result, positions_gt):

    r = len(positions_result)
    c = len(positions_gt)
    cost_matrix = np.ones((r,c))
    for i, positions_result in enumerate(positions_result):
        for j, position_gt in enumerate(positions_gt):
            cost_matrix[i,j] = compute_dis(positions_result, position_gt)
    return cost_matrix

def compute_metric_step(positions_result, positions_gt, id_history=None):
    mytresh = 0.4
    cost_matrix = compute_cost_martix_position(positions_result, positions_gt)
    matches, unmatched_result, unmatched_gt = linear_assignment(cost_matrix, thresh=mytresh)
    count_successed = len(matches)
    count_all = len(unmatched_result)+len(unmatched_gt)+count_successed
    dis = (len(unmatched_result) + len(unmatched_gt)) * mytresh
    #dis = 0
    for id_position_result, id_position_gt in matches:
        dis = compute_dis(positions_result[id_position_result], positions_gt[id_position_gt])+dis
    # if a target should be found but not be found, id_switch+1
    # if a target is false target, id_switch+1
    id_switch_step = len(unmatched_gt) + len(unmatched_result)
    #id_switch_step = 0
    if id_history is None:
        id_history = {}
        for [i, j] in matches:
            id_history[positions_gt[j][0]] = positions_result[i][0]
            #id_history[positions_gt[j][0]] = 0
        for j in unmatched_gt:
            id_history[positions_gt[j][0]] = -1
    else:
        for [i, j] in matches:
            if id_history[positions_gt[j][0]] != -1:
                if id_history[positions_gt[j][0]] != positions_result[i][0]:
                #if id_history[positions_gt[j][0]] != 0:
                    id_switch_step = id_switch_step + 1
            id_history[positions_gt[j][0]] = positions_result[i][0]
    return count_successed, count_all, dis, id_switch_step, id_history

def read(path):
    f = open(path)
    datas = f.readlines()
    out = []
    for data in datas:
        data = data.strip('\n').split(',')
        try:
            data = [float(item) for item in data]
        except:
            for i in range(1, len(data)):
               data[i] =  float(data[i])
        #data[-1] = int(data[-1])
        out.append(data)
    return out

def compute_metric_sequence(path_result_sequence, path_gt_sequence):
    count_successed_sequence = 0
    count_all_sequence = 0
    dis_sequence = 0
    id_switch_sequence = 0
    id_history=None
    files_result = os.listdir(path_result_sequence)
    files_gt = os.listdir(path_gt_sequence)
    files_result.sort()
    files_gt.sort()
    for step, file_gt in enumerate(files_gt):
        positions_result = read(path_result_sequence+files_result[step])
        positions_gt = read(path_gt_sequence+files_gt[step])
        count_successed_step, count_all_step, dis_step, id_switch_step, id_history = compute_metric_step(positions_result, positions_gt, id_history)
        count_successed_sequence = count_successed_sequence+ count_successed_step
        count_all_sequence = count_all_sequence + count_all_step
        dis_sequence = dis_sequence + dis_step
        id_switch_sequence = id_switch_sequence + id_switch_step
    return  count_successed_sequence, count_all_sequence, dis_sequence, id_switch_sequence

def compute_metric_dataset(path_result, path_gt):
    count_successed_dataset = 0
    count_all_dataset = 0
    dis_dataset = 0
    id_switch = 0

    sequences = os.listdir(path_result)
    sequences.sort()
    for sequence in sequences:
        #sequence = '46'
        print('sequence is : %s'%(sequence))
        path_result_sequence = path_result + sequence + '/'
        path_gt_sequence = path_gt + sequence + '/'
        count_successed_sequence, count_all_sequence, dis_sequence, id_switch_sequence = compute_metric_sequence(path_result_sequence, path_gt_sequence)
        print(count_successed_sequence/count_all_sequence, dis_sequence/count_all_sequence, id_switch_sequence)
        count_successed_dataset = count_successed_dataset + count_successed_sequence
        count_all_dataset = count_all_dataset + count_all_sequence
        dis_dataset = dis_dataset + dis_sequence
        id_switch = id_switch + id_switch_sequence

    precision = count_successed_dataset*1.0 / count_all_dataset
    distance = dis_dataset / count_successed_dataset
    return precision, distance, id_switch



if __name__ == '__main__':
    path_result = './result/'
    path_gt = './gt/'

    precision, distance, id_switch = compute_metric_dataset(path_result, path_gt)
    print(precision, distance, id_switch)









