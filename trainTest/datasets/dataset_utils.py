import os
import math
import torch
import numpy as np
from trainTest.datasets.dataloader_shffule_utils import initDatasetShffule
from utilss.common_utils import calculate_class_weights_torch
from sklearn.neighbors import NearestNeighbors
from models.supervisedModels.GNN_utils import trans_adj_to_edge_index_pcc_knn


def get_fileName_weights(path, subject, subjects_list):
    if subject not in subjects_list:
        raise ValueError('subject not in subjects_list_global', subjects_list)
    encoded_label_name = 'sub_label_encoded'
    file_name = os.path.join(path, ''.join([subject, '_phase_angle_targetTrainData.npz']))
    with open(file_name, 'rb') as f:
        raw_labels = np.load(f)[encoded_label_name]
    raw_label_type = list(np.unique(raw_labels[:,0]))
    class_weights = calculate_class_weights_torch(raw_labels[:,0])

    return file_name,  class_weights ,encoded_label_name, raw_label_type


def get_intra_dataloaders(path, label_name, total_exp_time, current_exp_time, train_batch, test_batch,
                          valid_batch, modal):
    init_dataset = initDatasetShffule()
    init_dataset.initIntraSubjectDataset(path=path, label_name=label_name, total_exp_time=total_exp_time, modal=modal)
    train_loader, valid_loader, test_loader = init_dataset.getDataLoader_intra(exp_time=current_exp_time,
                                                                               train_batch=train_batch,
                                                                               test_batch=test_batch,
                                                                               valid_batch=valid_batch)
    return train_loader, valid_loader, test_loader


def get_pcc_knn_edge_index_from_dataloader(dataloader, params, edge_gen_mode):
    if edge_gen_mode == 'AWMF':
        pcc_knn_adjs = get_pcc_knn_adj_from_dataloader(dataloader, params, edge_gen_mode)
        pcc_adj = pcc_knn_adjs['pcc_adj']
        knn_adj = pcc_knn_adjs['knn_adj']
        pcc_edge_index = trans_adj_to_edge_index_pcc_knn(pcc_adj)
        knn_edge_index = trans_adj_to_edge_index_pcc_knn(knn_adj)
        pcc_knn_edge_index = {'pcc_edge_index': pcc_edge_index, 'knn_edge_index': knn_edge_index}
    elif edge_gen_mode == 'PCC':
        pcc_knn_adjs = get_pcc_knn_adj_from_dataloader(dataloader, params, edge_gen_mode)
        pcc_adj = pcc_knn_adjs['pcc_adj']
        pcc_edge_index = trans_adj_to_edge_index_pcc_knn(pcc_adj)
        pcc_knn_edge_index = {'pcc_edge_index': pcc_edge_index}
    elif edge_gen_mode == 'KNN':
        pcc_knn_adjs = get_pcc_knn_adj_from_dataloader(dataloader, params, edge_gen_mode)
        knn_adj = pcc_knn_adjs['knn_adj']
        knn_edge_index = trans_adj_to_edge_index_pcc_knn(knn_adj)
        pcc_knn_edge_index = {'knn_edge_index': knn_edge_index}
    else:
        pcc_knn_edge_index = None
    return pcc_knn_edge_index


def get_pcc_knn_adj_from_dataloader(dataloader, params, edge_gen_mode):
    all_pcc_matrix = []
    all_knn_matrix = []
    pcc_act_thr = params['pcc_act_thr']
    kgnn_ratio = params['kgnn_ratio']
    vote_ratio = params['vote_ratio']
    if edge_gen_mode == 'AWMF':
        for data, target in dataloader:
            # [batch, 1, 15, 96]
            batch_data = data.to(device='cpu')
            # [batch, 15, 96]
            batch_data = torch.squeeze(batch_data, dim=1).numpy()
            # [batch, 15, 15]
            batch_pcc_matrix = generate_batch_pcc_matrix(batch_data, pcc_act_thr)
            # [batch, 15, 15]
            batch_knn_matrix = generate_batch_knn_matrix(batch_data, kgnn_ratio)
            all_pcc_matrix.extend(batch_pcc_matrix)
            all_knn_matrix.extend(batch_knn_matrix)
        # all_pcc_matrix / all_knn_matrix: [samplesize, 15, 15]
        # pcc_adj / knn_adj : [15, 15]
        pcc_adj = np.mean(all_pcc_matrix, axis=0)
        knn_adj = np.mean(all_knn_matrix, axis=0)
        # 激活操作
        if 0 <= vote_ratio <= 1:
            act_pcc_adj = np.where(pcc_adj >= vote_ratio, 1, 0).astype('int')
            act_knn_adj = np.where(knn_adj >= vote_ratio, 1, 0).astype('int')
        else:
            raise ValueError('pcc_act_thr or kgnn_act_thr must be in [0, 1]')
        pcc_adj, knn_adj = torch.from_numpy(act_pcc_adj), torch.from_numpy(act_knn_adj)
        pcc_knn_adjs = {'pcc_adj': pcc_adj, 'knn_adj': knn_adj}
    elif edge_gen_mode == 'PCC':
        for data, target in dataloader:
            batch_data = data.to(device='cpu')
            batch_data = torch.squeeze(batch_data, dim=1).numpy()
            batch_pcc_matrix = generate_batch_pcc_matrix(batch_data, pcc_act_thr)
            all_pcc_matrix.extend(batch_pcc_matrix)
        pcc_adj = np.mean(all_pcc_matrix, axis=0)
        if 0 <= vote_ratio <= 1:
            act_pcc_adj = np.where(pcc_adj >= pcc_act_thr, 1, 0).astype('int')
        else:
            raise ValueError('pcc_act_thr must be in [0, 1]')
        pcc_adj = torch.from_numpy(act_pcc_adj)
        pcc_knn_adjs = {'pcc_adj': pcc_adj}
    else:
        for data, target in dataloader:
            batch_data = data.to(device='cpu')
            batch_data = torch.squeeze(batch_data, dim=1).numpy()
            batch_knn_matrix = generate_batch_knn_matrix(batch_data, kgnn_ratio)
            all_knn_matrix.extend(batch_knn_matrix)
        knn_adj = np.mean(all_knn_matrix, axis=0)
        if 0 <= vote_ratio <= 1:
            act_knn_adj = np.where(knn_adj >= vote_ratio, 1, 0).astype('int')
        else:
            raise ValueError('kgnn_act_thr must be in [0, 1]')
        knn_adj = torch.from_numpy(act_knn_adj)
        pcc_knn_adjs = {'knn_adj': knn_adj}
    return pcc_knn_adjs


def generate_batch_pcc_matrix(batch_data, pcc_act_thr):
    batch_pcc_matrix = []
    for i in range(batch_data.shape[0]):
        # [1, 15, 96]
        data = batch_data[i, :, :]
        # [15, 96]
        # print(data.shape)
        # data = np.squeeze(data, axis=0)
        # [15, 15]
        pcc_matrix = np.abs(np.corrcoef(data))
        pcc_matrix = np.where(pcc_matrix >= pcc_act_thr, 1, 0).astype('int')
        # [batch, 15, 15]
        batch_pcc_matrix.append(pcc_matrix)

    return np.array(batch_pcc_matrix)


def generate_batch_knn_matrix(batch_data, kgnn_ratio):
    node_amount = batch_data.shape[1]
    k_neighbors = math.floor(kgnn_ratio * node_amount)
    if k_neighbors < 1:
        raise ValueError('k_neighbors < 1, please increase kgnn_ratio')
    elif k_neighbors > node_amount:
        raise ValueError('k_neighbors > node_amount, please decrease kgnn_ratio')
    else:
        batch_knn_matrix = []
        for i in range(batch_data.shape[0]):
            # [1, 15, 96]
            data = batch_data[i, :, :]
            # [15, 96]
            # data = np.squeeze(data, axis=0)
            knn_graph = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean')
            knn_graph.fit(data)
            distances, indices = knn_graph.kneighbors(data)
            # 构建KNN图 [15, 15]
            knn_matrix = np.zeros((node_amount, node_amount), dtype=int)
            for k in range(node_amount):
                for neighbor in indices[k]:
                    knn_matrix[k][neighbor] = 1
            batch_knn_matrix.append(knn_matrix)

        return np.array(batch_knn_matrix)


def get_print_info(subjects_list):
    info = ['当前任务：下肢运动识别，总受试者：', subjects_list]
    return info


def get_save_path(base_path, model_name, subject):
    absolute_path = os.path.join(base_path, model_name, ''.join(['Sub', subject]))
    relative_path = os.path.relpath(absolute_path, base_path)
    return {'absolute_path': absolute_path, 'relative_path': relative_path}


def get_basic_node_amount(modal):
    if modal == 'E':
        basic_node_amount = 14
    elif modal == 'A':
        basic_node_amount = 15
    elif modal == 'G':
        basic_node_amount = 4
    elif modal == 'E-G':
        basic_node_amount = 18
    elif modal == 'E-A':
        basic_node_amount = 29
    else:
        raise ValueError('modal error!')
    return basic_node_amount


def get_basic_node_amount_tj(modal):
    if modal == 'E':
        basic_node_amount = 16
    elif modal == 'A':
        basic_node_amount = 15
    elif modal == 'G':
        basic_node_amount = 7
    elif modal == 'E-A':
        basic_node_amount = 31
    elif modal == 'E-G':
        basic_node_amount = 23
    else:
        raise ValueError('modal error!')
    return basic_node_amount
