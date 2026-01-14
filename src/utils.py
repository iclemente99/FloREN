from pyHGT.data import Graph
import numpy as np
import torch
from collections import defaultdict
#import resource
import pandas as pd
try:
    import resource
except ImportError:
    resource = None

#def debuginfoStr(info):
#    print(info)
#    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024*1024)
#    print('Mem consumption (GB): '+str(mem))
def debuginfoStr(info):
    print(info)
    try:
        if resource is None:
            return
        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
        print('Mem consumption (GB): ' + str(mem))
    except Exception:
        # Silently skip on Windows or any unexpected platform issues
        pass


def loadGAS(data_path):
    df=pd.read_csv(data_path, sep=" ")
    return df.to_numpy()



def build_graph(gene_cell,tfs_f,cells_f,encoded,encoded2):
    g_index,c_index = np.nonzero(gene_cell)
    # 加上偏移量作为cell的节点标号
    c_index += gene_cell.shape[0]
    edges=torch.tensor([g_index, c_index], dtype=torch.float)
    
    # 这里是直接对graph.edge_list进行修改了，不是副本
    graph = Graph()
    s_type, r_type, t_type = ('gene', 'g_c', 'cell')
    elist = graph.edge_list[t_type][s_type][r_type]
    rlist = graph.edge_list[s_type][t_type]['rev_' + r_type]
    year = 1
    for s_id, t_id in edges.t().tolist():
        elist[t_id][s_id] = year
        rlist[s_id][t_id] = year

    g1_index,g2_index = np.nonzero(tfs_f)
    #g2_index += tfs_f.shape[0]
    edges_g = torch.tensor([g1_index, g2_index], dtype=torch.float)
    s_type, g_type, t_type = ('gene', 'g_g', 'gene')
    tlist = graph.edge_list[s_type][t_type][g_type]
    trlist = graph.edge_list[s_type][t_type]['rev_' + g_type]
    year = 1
    for s_id, t_id in edges_g.t().tolist():
        tlist[t_id][s_id] = year
        trlist[s_id][t_id] = year
        
    c1_index,c2_index = np.nonzero(cells_f)
    c1_index += gene_cell.shape[0]
    c2_index += gene_cell.shape[0]
    edges_c = torch.tensor([c1_index, c2_index], dtype=torch.float)
    s_type, c_type, t_type = ('cell', 'c_c', 'cell')
    clist = graph.edge_list[s_type][t_type][c_type]
    crlist = graph.edge_list[s_type][t_type]['rev_' + c_type]
    year = 1
    for s_id, t_id in edges_c.t().tolist():
        clist[t_id][s_id] = year
        crlist[s_id][t_id] = year
        
    #print('gene matrix: ',encoded.shape)
    #print('cell matrix: ',encoded2.shape)
    graph.node_feature['gene'] = torch.tensor(encoded, dtype=torch.float)
    graph.node_feature['cell'] = torch.tensor(encoded2, dtype=torch.float)

    graph.years = np.ones(gene_cell.shape[0]+gene_cell.shape[1])
    return graph

def build_data(adj, encoded, encoded2,tfs_f, cells_f,edge_dict):
    node_type = [0]*adj.shape[0]+[1]*adj.shape[1]
    node_type = torch.LongTensor(node_type)
    
    g_index,c_index = np.nonzero(adj)
    c_index += adj.shape[0]
    edge_index = torch.tensor([g_index, c_index], dtype=torch.long)
    edge_type = torch.LongTensor([edge_dict['g_c']]*edge_index.shape[1])
    edge_time = torch.LongTensor([0]*edge_index.shape[1])
    
    g1_index,g2_index = np.nonzero(tfs_f)
    #c_index += adj.shape[0]
    edge_index_g = torch.tensor([g1_index, g2_index], dtype=torch.long)
    edge_type_g = torch.LongTensor([edge_dict['g_g']]*edge_index_g.shape[1])
    edge_time_g = torch.LongTensor([0]*edge_index_g.shape[1])
    
    c1_index,c2_index = np.nonzero(cells_f)
    c1_index += adj.shape[0]
    c2_index += adj.shape[0]
    edge_index_c = torch.tensor([c1_index, c2_index], dtype=torch.long)
    edge_type_c = torch.LongTensor([edge_dict['c_c']]*edge_index_c.shape[1])
    edge_time_c = torch.LongTensor([0]*edge_index_c.shape[1])
    
    edge_index = torch.tensor([np.concatenate((g_index,g1_index,c1_index)),np.concatenate((c_index,g2_index,c2_index))], dtype=torch.long)
    edge_type = torch.cat((edge_type,edge_type_g,edge_type_c))
    edge_time = torch.cat((edge_time,edge_time_g,edge_time_c))
    
    x = {'gene': torch.tensor(encoded, dtype=torch.float),
         'cell': torch.tensor(encoded2, dtype=torch.float)} 
    # print(len(x['gene']))  # 5000
    # print(len(x['cell']))  # 2713
    # print(len(node_type))  # 7713
    
    return x,node_type, edge_time, edge_index,edge_type


