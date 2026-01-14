from .conv import *
from .mlp import *
#from .sub_sample import sub_sample
from .layers import AvgReadout, Discriminator, Discriminator2
import numpy as np

class Classifier(nn.Module):
    def __init__(self, n_hid, n_out):
        super(Classifier, self).__init__()
        self.n_hid    = n_hid
        self.n_out    = n_out
        self.linear   = nn.Linear(n_hid,  n_out)
    def forward(self, x):
        tx = self.linear(x)
        return torch.log_softmax(tx.squeeze(), dim=-1)
    def __repr__(self):
        return '{}(n_hid={}, n_out={})'.format(
            self.__class__.__name__, self.n_hid, self.n_out)

class Matcher(nn.Module):
    '''
        Matching between a pair of nodes to conduct link prediction.
        Use multi-head attention as matching model.
    '''
    def __init__(self, n_hid):
        super(Matcher, self).__init__()
        self.left_linear    = nn.Linear(n_hid,  n_hid)
        self.right_linear   = nn.Linear(n_hid,  n_hid)
        self.sqrt_hd  = math.sqrt(n_hid)
        self.cache      = None
    def forward(self, x, y, infer = False, pair = False):
        ty = self.right_linear(y)
        if infer:
            '''
                During testing, we will consider millions or even billions of nodes as candidates (x).
                It's not possible to calculate them again for different query (y)
                Since the model is fixed, we propose to cache them, and dirrectly use the results.
            '''
            if self.cache != None:
                tx = self.cache
            else:
                tx = self.left_linear(x)
                self.cache = tx
        else:
            tx = self.left_linear(x)
        if pair:
            res = (tx * ty).sum(dim=-1)
        else:
            res = torch.matmul(tx, ty.transpose(0,1))
        return res / self.sqrt_hd
    def __repr__(self):
        return '{}(n_hid={})'.format(
            self.__class__.__name__, self.n_hid)
    


        
#class GNN_class(nn.Module):
#    def __init__(self, in_dim, n_hid, h_out, num_types, num_relations, n_heads, n_layers, dropout = 0.2, conv_name = 'hgt', prev_norm = True, last_norm = True, use_RTE = True):
#        super(GNN, self).__init__()
#        self.gcs = nn.ModuleList()
#        self.num_types = num_types
#        self.in_dim    = in_dim
#        self.n_hid     = n_hid
#        self.adapt_ws  = nn.ModuleList()
#        self.drop      = nn.Dropout(dropout)
#        self.att =None
#        self.emb =None
#        self.conv_name = conv_name
#        self.mlp = Classifier(n_hid, h_out)
#        self.subsample = sub_sample()
#        for t in range(num_types):
#            self.adapt_ws.append(nn.Linear(in_dim, n_hid))
#        for l in range(n_layers - 1):
#            self.gcs.append(GeneralConv(conv_name, n_hid, n_hid, num_types, num_relations, n_heads, dropout, use_norm = prev_norm, use_RTE = use_RTE))
#        self.gcs.append(GeneralConv(conv_name, n_hid, n_hid, num_types, num_relations, n_heads, dropout, use_norm = last_norm, use_RTE = use_RTE))
#
#    def forward(self, node_feature, node_type, edge_time, edge_index, edge_type):
#        res = torch.zeros(node_feature.size(0), self.n_hid).to(node_feature.device)
#        #res = torch.zeros(node_feature[0].shape[0], self.n_hid).to(node_feature[0].device)
#        for t_id in range(self.num_types):
#            idx = (node_type == int(t_id))
#            if idx.sum() == 0:
#                continue
#            res[idx] = torch.tanh(self.adapt_ws[t_id](node_feature[idx]))
#        meta_xs = self.drop(res)
#        del res
#        self.att = {}
#        i=0
#        self.emb={}
#        for gc in self.gcs:
#            meta_xs = gc(meta_xs, node_type, edge_index, edge_type, edge_time)
#            if (self.conv_name == 'hgt'):
#                #self.att = gc.res_att
#                self.att[i]=gc.res_att
#                #self.emb[i]=gc.res
#                i=i+1
#                #print(gc.res_att)
#                #for p in gc.parameters():
#                #    print(p)                
#        #self.att = gc.res_att
#        self.att = self.att[0]
#        cell_class = self.mlp(meta_xs[node_type==1,:])
#        return meta_xs, cell_class
#        #return meta_xs 

class GNN(nn.Module):
    def __init__(self, in_dim, n_hid, num_types, num_relations, n_heads, n_layers, n_labels, dropout = 0.2, conv_name = 'hgt', prev_norm = True, last_norm = True, use_RTE = True):
        super(GNN, self).__init__()
        self.gcs = nn.ModuleList()
        self.num_types = num_types
        self.in_dim    = in_dim
        self.n_hid     = n_hid
        self.adapt_ws  = nn.ModuleList()
        self.drop      = nn.Dropout(dropout)
        self.att =None
        self.emb_104 =None
        self.emb =None
        self.att_1 = None
        self.att_2 = None
        self.att_0 = None
        self.att_n = None
        self.cells_1 = None
        self.cells_2 = None
        self.cells_0 = None
        self.cells_n = None
        self.genes_1 = None
        self.genes_2 = None
        self.genes_0 = None
        self.genes_n = None
        self.g_32 = None
        self.l_32 = None
        self.g_64 = None
        self.l_64 = None
        self.g_128 = None
        self.l_128 = None
        self.conv_name = conv_name
        for t in range(num_types):
            self.adapt_ws.append(nn.Linear(in_dim, n_hid))
        for l in range(n_layers - 1):
            self.gcs.append(GeneralConv(conv_name, n_hid, n_hid, num_types, num_relations, n_heads, dropout, use_norm = prev_norm, use_RTE = use_RTE))
        self.gcs.append(GeneralConv(conv_name, n_hid, n_hid, num_types, num_relations, n_heads, dropout, use_norm = last_norm, use_RTE = use_RTE))
        self.mlp = MLP(n_hid,n_labels)
        #self.subsample = sub_sample()
        self.disc = Discriminator(n_hid)
        self.disc2 = Discriminator2(n_hid)
        self.read = AvgReadout()
        
    def forward_one(self, node_feature, node_type, edge_time, edge_index, edge_type):
        # type-specific projection
        res = torch.zeros(node_feature.size(0), self.n_hid).to(node_feature.device)
        for t_id in range(self.num_types):
            idx = (node_type == int(t_id))
            if idx.sum() == 0:
                continue
            #res[idx] = torch.tanh(self.adapt_ws[t_id](node_feature[idx]))
            #res[idx] = torch.nn.functional.relu(self.adapt_ws[t_id](node_feature[idx]))
            #res[idx] = torch.nn.functional.silu(self.adapt_ws[t_id](node_feature[idx]))
            #res[idx] = torch.nn.functional.prelu(self.adapt_ws[t_id](node_feature[idx]))
            #res[idx] = torch.nn.functional.leaky_relu(self.adapt_ws[t_id](node_feature[idx]))
            res[idx] = self.adapt_ws[t_id](node_feature[idx])
        
        meta_xs = self.drop(res)
        del res
        # go through stacked HGT layers
        self.att = {}
        self.emb = {}
        i = 0
        for gc in self.gcs:
            meta_xs = gc(meta_xs, node_type, edge_index, edge_type, edge_time)
            if self.conv_name == "hgt":
                self.att[i] = gc.res_att
            self.emb[i] = gc.res
            i += 1
        
        return meta_xs, self.att[0]
        
    def forward(self, node_feature_0, node_type_0, edge_time_0, edge_index_0, edge_type_0, node_feature_1, node_type_1, edge_time_1, edge_index_1, edge_type_1, node_feature_2, node_type_2, edge_time_2, edge_index_2, edge_type_2):
        #res = torch.zeros(node_feature.size(0), self.n_hid).to(node_feature.device)
        #res = torch.zeros(node_feature[0].shape[0], self.n_hid).to(node_feature[0].device)
        #for t_id in range(self.num_types):
        #    idx = (node_type == int(t_id))
        #    if idx.sum() == 0:
        #        continue
        #    res[idx] = torch.tanh(self.adapt_ws[t_id](node_feature[idx]))
        #meta_xs = self.drop(res)
        #del res
        #self.att = {}
        #i=0
        #self.emb={}
        #for gc in self.gcs:
        #    meta_xs = gc(meta_xs, node_type, edge_index, edge_type, edge_time)
        #    if (self.conv_name == 'hgt'):
        #        #self.att = gc.res_att
        #        self.att[i]=gc.res_att
        #        #self.emb[i]=gc.res
        #        i=i+1
        #        #print(gc.res_att)
        #        #for p in gc.parameters():
        #        #    print(p)                
        #self.att = gc.res_att
        meta_xs_1, self.att_1 = self.forward_one(node_feature_1, node_type_1, edge_time_1, edge_index_1, edge_type_1)
        #print(meta_xs_1.shape)
        meta_xs_2, self.att_2 = self.forward_one(node_feature_2, node_type_2, edge_time_2, edge_index_2, edge_type_2)
        meta_xs_0, self.att_0 = self.forward_one(node_feature_0, node_type_0, edge_time_0, edge_index_0, edge_type_0)
        idx = torch.randperm(node_feature_0.size(0), device=node_feature_0.device)
        node_feature_n = node_feature_0[idx,:]
        meta_xs_n, self.att_n = self.forward_one(node_feature_n, node_type_0, edge_time_0, edge_index_0, edge_type_0)
        #print(meta_xs_2.shape)
        #print(meta_xs_0.shape)
        #print(meta_xs_n.shape)
        #self.att_1 = self.att[0]
        self.cells_1 = meta_xs_1[node_type_1 == 1,:]
        self.genes_1 = meta_xs_1[node_type_1 == 0,:]
        #self.att_2 = self.att[0]
        self.cells_2 = meta_xs_2[node_type_2 == 1,:]
        self.genes_2 = meta_xs_2[node_type_2 == 0,:]
        self.cells_0 = meta_xs_0[node_type_0 == 1, :]
        self.genes_0 = meta_xs_0[node_type_0 == 0, :]
        self.cells_n = meta_xs_n[node_type_0 == 1, :]
        self.genes_n = meta_xs_n[node_type_0 == 0, :]
        # Add batch dimension to convert [num_nodes, n_h] to [1, num_nodes, n_h]
        meta_xs_1 = meta_xs_1.unsqueeze(0)  # From [2405, 104] to [1, 2405, 104]
        meta_xs_2 = meta_xs_2.unsqueeze(0)  # From [2405, 104] to [1, 2405, 104]
        meta_xs_0 = meta_xs_0.unsqueeze(0)  # From [5981, 104] to [1, 5981, 104]
        meta_xs_n = meta_xs_n.unsqueeze(0)  # From [5981, 104] to [1, 5981, 104]
        c_1 = self.read(meta_xs_1, None)
        #print(c_1.shape)
        c_2 = self.read(meta_xs_2, None)
        #print(c_2.shape)
        ret1 = self.disc(c_1, meta_xs_0, meta_xs_n)
        #print(ret1.shape)
        ret2 = self.disc(c_2, meta_xs_0, meta_xs_n)
        #print(ret2.shape)
        ret = ret1 + ret2
        c_vector = np.vstack((c_1.cpu().detach().numpy(), c_2.cpu().detach().numpy()))
        #print(c_vector.shape)
        #ci_vector = np.hstack((c_1_i.detach().numpy(), c_3_i.detach().numpy()))
        cat_head = torch.FloatTensor(np.mean(c_vector, axis=0))
        #print((cat_head.max()-cat_head.min()))
        #print(cat_head.mean())
        #print(cat_head.shape)
        self.emb_104 = cat_head
        classifier, g_32, l_32, g_64, l_64, g_128, l_128 = self.mlp(cat_head)
        self.g_32 = g_32
        self.l_32 = l_32
        self.g_64 = g_64
        self.l_64 = l_64
        self.g_128 = g_128
        self.l_128 = l_128
        return ret, classifier

class GNN_from_raw(nn.Module):
    def __init__(self, in_dim, n_hid, num_types, num_relations, n_heads, n_layers, \
        dropout = 0.2, conv_name = 'hgt', \
        prev_norm = True, last_norm = True, \
        use_RTE = True,\
        AEtype=0\
        ):
        super(GNN_from_raw, self).__init__()
        self.gcs = nn.ModuleList()
        self.num_types = num_types
        self.in_dim    = in_dim
        self.n_hid     = n_hid
        self.adapt_ws  = nn.ModuleList()
        self.drop      = nn.Dropout(dropout)
        self.embedding1 = nn.ModuleList()
        self.embedding2 = nn.ModuleList()
        self.decode1 = nn.ModuleList()
        self.decode2 = nn.ModuleList()
        self.AEtype = AEtype
        self.att =None
        self.conv_name = conv_name
        for ti in range(num_types):
             #self.embedding.append(F.relu(nn.Linear(512,256)(F.relu(nn.Linear(in_dim[ti],512)))))
             self.embedding1.append(nn.Linear(in_dim[ti],512)) #embedding1[0] [2713 x 512] embedding1[1] [24022 x 512]
             self.embedding2.append(nn.Linear(512,256)) #embedding2[0] [512, 256] embedding2[1] [512,256]
        
        if AEtype==1: #embedding autoencoder
           for ti in range(num_types):
                #self.embedding.append(F.relu(nn.Linear(512,256)(F.relu(nn.Linear(in_dim[ti],512)))))
                self.decode1.append(nn.Linear(256,512)) #embedding1[0] [2713 x 512] embedding1[1] [24022 x 512]
                self.decode2.append(nn.Linear(512,in_dim[ti])) #embedding2[0] [512, 256] embedding2[1] [512,256]
        elif AEtype==2:
            for ti in range(num_types):
                #self.embedding.append(F.relu(nn.Linear(512,256)(F.relu(nn.Linear(in_dim[ti],512)))))
                self.decode1.append(nn.Linear(n_hid,512)) #embedding1[0] [2713 x 512] embedding1[1] [24022 x 512]
                self.decode2.append(nn.Linear(512,in_dim[ti])) #embedding2[0] [512, 256] embedding2[1] [512,256]
        
        for t in range(num_types):
            self.adapt_ws.append(nn.Linear(256, n_hid)) #256 could be one additional hyperparameter!!!
        for l in range(n_layers - 1):
            self.gcs.append(GeneralConv(conv_name, n_hid, n_hid, num_types, num_relations, n_heads, dropout, use_norm = prev_norm, use_RTE = use_RTE))
        self.gcs.append(GeneralConv(conv_name, n_hid, n_hid, num_types, num_relations, n_heads, dropout, use_norm = last_norm, use_RTE = use_RTE))
    
    
    def encode(self, x,t_id):
        h1 = F.relu(self.embedding1[t_id](x))
        return F.relu(self.embedding2[t_id](h1))
    
    def decode(self, z,t_id):
        h3 = F.relu(self.decode1[t_id](z))
        return torch.relu(self.decode2[t_id](h3))
        #return torch.relu(self.fc4(z))
    
    def forward(self, node_feature, node_type, edge_time, edge_index, edge_type):
        node_embedding=[] #len = 2 
        for t_id in range(self.num_types):
            node_embedding += list(self.encode(node_feature[t_id],t_id))
        
        node_embedding_stack = torch.stack(node_embedding)
        #print("shape of node_embedding="+str(node_embedding_stack.shape)+"\n")
        res = torch.zeros(node_embedding_stack.size(0), self.n_hid).to(node_feature[0].device)
        
        for t_id in range(self.num_types):
            idx = (node_type == int(t_id)) #0, 1
            if idx.sum() == 0:
                continue
            #res[idx] = torch.tanh(self.adapt_ws[t_id](self.embedding[t_id](node_feature[idx])))
            #print(idx)
            res[idx] = torch.tanh(self.adapt_ws[t_id](node_embedding_stack[idx]))
            #res[idx] = torch.tanh(self.adapt_ws[t_id](self.encode(node_feature[t_id],t_id)))
        
        meta_xs = self.drop(res)
        del res
        for gc in self.gcs:
            meta_xs = gc(meta_xs, node_type, edge_index, edge_type, edge_time)
        if (self.conv_name == 'hgt'):
            self.att = gc.res_att    
        if self.AEtype!=0:
            if self.AEtype==1:#embedding auto-encoder
                 decode_embedding=[]
                 for t_id in range(self.num_types):
                       decode_embedding.append(self.decode(node_embedding_stack[node_type==t_id],t_id)) #0 genematrix 1 cellmatrix
                       #print(decode_embedding[t_id].shape)
                       #print(meta_xs[node_type==t_id].shape)
                 
                 return meta_xs,decode_embedding
            
            elif self.AEtype==2: #HGT embedding auto-encotder 
                 decode_embedding = []
                 for t_id in range(self.num_types):
                       decode_embedding.append(self.decode(meta_xs[node_type==t_id],t_id)) #0 genematrix 1 cellmatrix
                       #print("in model decode_embedding shape tid="+str(t_id))
                       #print(decode_embedding[t_id].shape)
                       #print(meta_xs[node_type==t_id].shape)
                 return meta_xs,decode_embedding
        else:
            return meta_xs  
