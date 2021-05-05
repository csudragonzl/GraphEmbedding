from ge import DeepWalk
import networkx as nx
import os
import tensorflow as tf
import pandas as pd
import numpy as np


def computePrecisionCurve(predicted_edge_list, true_digraph, max_k=-1):
    if max_k == -1:
        max_k = len(predicted_edge_list)
    else:
        max_k = min(max_k, len(predicted_edge_list))

    sorted_edges = sorted(predicted_edge_list, key=lambda x: x[2], reverse=True)

    precision_scores = []
    delta_factors = []
    correct_edge = 0
    for i in range(max_k):
        if true_digraph.has_edge(sorted_edges[i][0], sorted_edges[i][1]):
            correct_edge += 1
            delta_factors.append(1.0)
        else:
            delta_factors.append(0.0)
        precision_scores.append(1.0 * correct_edge / (i + 1))
    return precision_scores, delta_factors


def computeMAP(predicted_edge_list, true_digraph, max_k=-1):
    node_num = true_digraph.number_of_nodes()
    node_edges = []
    for i in range(node_num):
        node_edges.append([])
    for edge in predicted_edge_list:
        node_edges[edge[0]].append(edge)
    node_AP = [0.0] * node_num
    count = 0
    for i in range(node_num):
        if true_digraph.degree(i) == 0:
            continue
        count += 1
        precision_scores, delta_factors = computePrecisionCurve(node_edges[i], true_digraph, max_k)
        precision_rectified = [p * d for p, d in zip(precision_scores, delta_factors)]
        if (sum(delta_factors) == 0):
            node_AP[i] = 0
        else:
            node_AP[i] = float(sum(precision_rectified) / sum(delta_factors))
    return sum(node_AP) / count


if __name__ == "__main__":
    data_list = ['cellphone', 'enron', 'enron_large', 'HS11', 'HS12', 'primary', 'workplace']
    embeddings_list = []
    graphs = []
    pred_edges_list = []
    for data in data_list:
        basepath = 'data/' + data
        edge_list_path = os.listdir(basepath)
        if 'enron' in basepath or 'enron_large' in basepath:
            edge_list_path.sort(key=lambda x: int(x[5:-6]))
        elif 'HS11' in basepath or 'primary' in basepath or 'workplace' in basepath or 'fbmessages' in basepath:
            edge_list_path.sort(key=lambda x: int(x[-5:-4]))
        elif 'HS12' in basepath:
            edge_list_path.sort(key=lambda x: int(x[-11:-10]))
        elif 'cellphone' in basepath:
            edge_list_path.sort(key=lambda x: int(x[9:-6]))
        node_num = 0
        edges_list = []
        for i in range(len(edge_list_path)):
            file = open(os.path.join(basepath, edge_list_path[i]), 'r')
            # 不同的数据文件分隔符不一样
            if 'primary' in basepath or 'workplace' in basepath:
                edges = list(y.split(' ')[:2] for y in file.read().split('\n'))[:-1]
            elif 'enron_large' in basepath:
                edges = list(y.split(' ')[:2] for y in file.read().split('\n'))
            else:
                edges = list(y.split('\t')[:2] for y in file.read().split('\n'))[:-1]
            for j in range(len(edges)):
                # 将字符的边转为int型
                edges[j] = list(int(z) - 1 for z in edges[j])

            # 去除重复的边
            edges = list(set([tuple(t) for t in edges]))
            edges_temp = []
            for j in range(len(edges)):
                # 去除反向的边和自环
                if [edges[j][1], edges[j][0]] not in edges_temp and edges[j][1] != edges[j][0]:
                    edges_temp.append(edges[j])
                # 找到节点数
                for z in edges[j]:
                    node_num = max(node_num, z)
            edges_list.append(edges_temp)
        node_num += 1
        for edges in edges_list:
            graph = nx.DiGraph()
            graph.add_nodes_from([i for i in range(node_num)])
            graph.add_edges_from(edges)
            graphs.append(graph)

            if 'enron_large' in basepath:
                model = DeepWalk(graph, walk_length=10, num_walks=80, workers=1)
                model.train(window_size=5, iter=3)
            else:
                model = DeepWalk(graph, walk_length=10, num_walks=800, workers=1)
                model.train(window_size=5, iter=3)
            embeddings = model.get_embeddings()
            embeddings_list.append(embeddings)
            pred_edges = []
            emb_matrix = np.zeros([node_num, 128])
            for node, emb in embeddings.items():
                node = int(node)
                emb = emb.reshape((1, emb.shape[0]))
                emb_matrix[node] = emb
            emb_matrix = tf.convert_to_tensor(emb_matrix)
            pra = tf.matmul(emb_matrix, tf.transpose(emb_matrix))
            pra2 = tf.nn.sigmoid(pra)
            with tf.Session() as sess:
                sess.run(pra)
                sess.run(pra2)
                pred_adj = pra2.eval()
            for j in range(pred_adj.shape[0]):
                for k in range(j + 1, pred_adj.shape[0]):
                    if pred_adj[j][k] > 0.5:
                        pred_edges.append([j, k, pred_adj[j][k]])
                        pred_edges.append([k, j, pred_adj[j][k]])
            pred_edges_list.append(pred_edges)
        MAP_list = []
        for i in range(len(pred_edges_list) - 1):
            MAP = computeMAP(pred_edges_list[i], graphs[i + 1])
            MAP_list.append(MAP)
            print('第' + str(i) + '-' + str(i + 1) + '个时间片的MAP值为' + str(MAP))
        MAP_list.append(np.mean(MAP_list))
        result = {'MAP值': MAP_list}
        label = []
        for i in range(len(MAP_list) - 1):
            row = '第' + str(i + 1) + '-' + str(i + 1) + '个时间片'
            label.append(row)
        label.append('mean_MAP')
        csv_path = 'result/' + data + '.csv'
        df = pd.DataFrame(result, index=label)
        df.to_csv(csv_path)
