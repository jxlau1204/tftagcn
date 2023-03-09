import torch 
import torch.nn as nn  
import torch_geometric.nn as pygnn



def build_intra_graph(dialog_lengths, to_future_link, to_past_link, speakers, device="cpu"): 
    # build intra graph to update node embeding from same model
    device = dialog_lengths.device
    batch_size = dialog_lengths.shape[0]
    graphs = []
    feats = []
    for i in range(batch_size):
        dialog_length = dialog_lengths[i]
        adj = torch.eye(dialog_length)
        for ii in range(dialog_length):
            adj[ii,ii+1:(min(ii+to_future_link+1, dialog_length))] = 1
            adj[ii,(max(ii-to_past_link, 0)):ii] = 1
        graphs.append(adj)
    all_adj = torch.zeros(dialog_lengths.sum(), dialog_lengths.sum())
    for i in range(batch_size):
        start = dialog_lengths[:i].sum()
        end = start + dialog_lengths[i]
        all_adj[start:end, start:end] = graphs[i]
    relation_type = (speakers.cpu().reshape(-1,1) @ speakers.cpu().reshape(1, -1))[torch.where(all_adj!=0)].to(device)   
    all_adj = torch.stack(torch.where(all_adj!=0),dim = 0).to(device)
    
    return all_adj, relation_type



class GCNModule(torch.nn.Module):
    def __init__(self, input_size, hidden_size, layers=2, dropout=0.6, act = nn.ReLU()) -> None:
        super(GCNModule, self).__init__()
        self.gconvs = nn.ModuleList()
        self.gconvs.append(pygnn.GCNConv(input_size, hidden_size))
        for i in range(layers-1):
            self.gconvs.append(pygnn.GCNConv(hidden_size, hidden_size))
        self.dropout = nn.Dropout(dropout)
        self.act = act
    def forward(self, x, edge_index):
        h = x
        # if self.residual:
        #     x_0 = self.lin_proj(x)
        for i, con in enumerate(self.gconvs):
            h = self.dropout(h)
            h = con(h, edge_index)
            h = self.act(h)
        out = h
        return out
class GraphConvModule(torch.nn.Module):
    def __init__(self, input_size, hidden_size, layers=2, dropout=0.6, act = nn.ReLU()) -> None:
        super().__init__()
        self.gconvs = nn.ModuleList()
        self.gconvs.append(pygnn.GraphConv(input_size, hidden_size))
        for i in range(layers-1):
            self.gconvs.append(pygnn.GraphConv(hidden_size, hidden_size))
        self.dropout = nn.Dropout(dropout)
        self.act = act
    def forward(self, x, edge_index):
        h = x
        # if self.residual:
        #     x_0 = self.lin_proj(x)
        for i, con in enumerate(self.gconvs):
            h = self.dropout(h)
            h = con(h, edge_index)
            h = self.act(h)
        out = h
        return out
class GATModule(torch.nn.Module):
    def __init__(self, input_size, hidden_size=64, layers=2, head_num=3, dropout=0.5, act = nn.ReLU()) -> None:
        super(GATModule, self).__init__()
        self.head_num = head_num
        self.gconvs = nn.ModuleList()
        self.gconvs.append(pygnn.GATConv(input_size, hidden_size, self.head_num, dropout=dropout))
        for i in range(layers-2):
            self.gconvs.append(pygnn.GATConv(hidden_size*self.head_num, hidden_size, self.head_num, dropout=dropout))
        self.gconvs.append(pygnn.GATConv(hidden_size*self.head_num, hidden_size, 1, dropout=dropout))
        self.act = act
    def forward(self, x, edge_index):
        out = x
        for gnn in self.gconvs:
            out = gnn(out, edge_index)
            out = self.act(out)
        return out
class DIAGCN(nn.Module):
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 num_layers, 
                 dropout, 
                 n_classes, 
                 type, 
                 to_future_link=4, 
                 to_past_link=0) -> None:
        super().__init__()
        self.to_future_link = to_future_link
        self.to_past_link = to_past_link
        self.rgnn = pygnn.RGCNConv(input_size, hidden_size, 2)
        
        self.gnn_layer = pygnn.GraphConv(
                in_channels= hidden_size,
                out_channels=hidden_size
            )
        self.classify = nn.Sequential(
            nn.Linear(hidden_size, n_classes)
        )
        self.tem_graphs = {}
        self.skip_con = nn.Linear(input_size, hidden_size)
    def forward(self, input, dialog_lengths, speakers):
        ## Speed up graph building
        graph_keys = "_".join([str(int(length_i)) for length_i in dialog_lengths.cpu()] + [str(int(speaker_i)) for speaker_i in speakers.cpu()])
        self.tem_graphs[graph_keys] = self.tem_graphs.get(graph_keys, build_intra_graph(dialog_lengths, self.to_future_link, self.to_past_link, speakers))
        
        edges = self.tem_graphs[graph_keys][0]
        relation_type = self.tem_graphs[graph_keys][1]
        x = self.rgnn(input, edges, relation_type)
        
        x = self.gnn_layer(x, edges)
        out = self.classify(x + self.skip_con (input))
        return out