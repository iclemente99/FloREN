import numpy as np
from collections import defaultdict

def norm_rowcol(matrix):
    # 按行求和
    row_norm=np.sum(matrix,axis=1).reshape(-1,1)
    # 行归一化
    matrix=matrix/row_norm
    # 按列求和
    col_norm=np.sum(matrix,axis=0)
    return matrix/col_norm

def sub_sample(graph,GAS, sampling_size,gene_size,gene_shape,cell_shape, query=False):
    cell_indexs=gene_shape+np.random.choice(np.arange(cell_shape),sampling_size,replace=False) # Generates sampling size number of random indexes for cells 
    sub_matrix=GAS[:,cell_indexs-gene_shape] # Substract a samplinz matrix with sampled cells
    if query == True:
        gene_indexs = np.arange(gene_size)
    else: 
        gene_indexs=np.nonzero(np.sum(sub_matrix,axis=1))[0] # Extracts genes indexes with connections on those cells
    
    sub_matrix=GAS[gene_indexs,:][:,cell_indexs-gene_shape] # Substracts a sampling matrix with genes connectec to those sampling cells
    
    #sub_matrix=norm_rowcol(sub_matrix) # Normalizes the matrix over genes anc ells connections
    
    _indexs=np.argsort(np.sum(sub_matrix,axis=1))[::-1] # Extracts gene indexes based in their connectivity to sampling cells
    gene_indexs=gene_indexs[_indexs] # Gets the genes out of the list of genes
    gene_indexs=gene_indexs[:gene_size] # Gets the sampling size number highest connected genes
    
    feature={
        'gene':graph.node_feature['gene'][gene_indexs,:],
        'cell':graph.node_feature['cell'][cell_indexs-gene_shape,:],
        #'cell':graph.node_feature['cell'][cell_indexs,:],
    }
    
    times={
        'gene': np.ones(gene_size),
        'cell':np.ones(sampling_size)
    }
    
    indxs={
        'gene':gene_indexs,
        'cell':cell_indexs-gene_shape
        #'cell':cell_indexs
    }
    
    edge_list = defaultdict(  # target_type
        lambda: defaultdict(  # source_type
            lambda: defaultdict(  # relation_type
                lambda: []  # [target_id, source_id]
            )))
    
    for i in range(gene_size):
        edge_list['gene']['gene']['self'].append([i,i])
    
    for i in range(sampling_size):
        edge_list['cell']['cell']['self'].append([i,i])
    
    for i,cell_id in enumerate(cell_indexs):
        for j,gene_id in enumerate(gene_indexs):
            if gene_id in graph.edge_list['cell']['gene']['g_c'][cell_id]:
                edge_list['cell']['gene']['g_c'].append([i,j])
                #edge_list['cell']['gene']['g_c'].append([i,j+gene_size])
                edge_list['gene']['cell']['rev_g_c'].append([j,i])
                #edge_list['gene']['cell']['rev_g_c'].append([j+gene_size,i])
                  
    for i,gene_id_i in enumerate(gene_indexs):
        for j,gene_id_ii in enumerate(gene_indexs):
            if gene_id_ii in graph.edge_list['gene']['gene']['g_g'][gene_id_i]:
                edge_list['gene']['gene']['g_g'].append([i,j])
                edge_list['gene']['gene']['rev_g_g'].append([j,i])
                
    for i,cell_id_i in enumerate(cell_indexs):
        for j,cell_id_ii in enumerate(cell_indexs):
            if cell_id_ii in graph.edge_list['cell']['cell']['c_c'][cell_id_i]:
                edge_list['cell']['cell']['c_c'].append([i,j])
                #edge_list['cell']['cell']['c_c'].append([i+gene_size,j+gene_size])
                edge_list['cell']['cell']['rev_c_c'].append([j,i])
                #edge_list['cell']['cell']['rev_c_c'].append([j+gene_size,i+gene_size])
                
    return feature, times, edge_list, indxs
