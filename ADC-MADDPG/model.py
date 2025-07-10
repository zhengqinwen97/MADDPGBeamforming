
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv


class abstract_agent(nn.Module):
    def __init__(self):
        super(abstract_agent, self).__init__()
    
    def act(self, input):
        policy, value = self.forward(input)
        return policy, value
    

class GraphEncoder(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim, num_nodes):
        super(GraphEncoder, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim

        self.edge_nn1 = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_feat_dim * hidden_dim)  # shape: [E, out × in]
        )
        self.conv1 = NNConv(in_channels=node_feat_dim, out_channels=hidden_dim, nn=self.edge_nn1, aggr='mean')

        self.edge_nn2 = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * hidden_dim)  # shape: [E, out × in]
        )
        self.conv2 = NNConv(in_channels=hidden_dim, out_channels=hidden_dim, nn=self.edge_nn2, aggr='mean')

    def forward(self, x, edge_index, edge_attr):
        h = F.relu(self.conv1(x, edge_index, edge_attr))  # [B, hidden]
        h = F.relu(self.conv2(h, edge_index, edge_attr))  # [B, hidden]
        h = h.view(1, self.num_nodes * self.hidden_dim)   # [1, B × hidden]
        return h
    

class openai_actor(abstract_agent):
    def __init__(self, node_feat_dim, edge_feat_dim, num_nodes, action_size, args):
        super(openai_actor, self).__init__()
        self.tanh = nn.Tanh()
        self.LReLU = nn.LeakyReLU(0.01)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

        self.encoder = GraphEncoder(node_feat_dim, edge_feat_dim, args.gnn_hidden_dim, num_nodes)
        graph_embed_dim = num_nodes * args.gnn_hidden_dim

        self.linear_a1 = nn.Linear(graph_embed_dim, args.num_units_1)
        self.linear_a2 = nn.Linear(args.num_units_1, args.num_units_2)
        self.linear_a3 = nn.Linear(args.num_units_2, args.num_units_2)
        self.linear_a = nn.Linear(args.num_units_2, action_size)
        self.linear_s = nn.Linear(args.num_units_2, 1)

        self.reset_parameters()
        self.train()
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        gain_tanh = nn.init.calculate_gain('tanh')
        nn.init.xavier_uniform_(self.linear_a1.weight, gain=gain)
        nn.init.xavier_uniform_(self.linear_a2.weight, gain=gain)
        nn.init.xavier_uniform_(self.linear_a3.weight, gain=gain)
        nn.init.xavier_uniform_(self.linear_a.weight, gain=gain_tanh)
        nn.init.xavier_uniform_(self.linear_s.weight, gain=gain_tanh)

    def forward(self, x, edge_index, edge_attr, model_original_out=False):
        # Encode graph
        graph_feat = self.encoder(x, edge_index, edge_attr)

        # Pass through MLP
        x = self.LReLU(self.linear_a1(graph_feat))
        x = self.LReLU(self.linear_a2(x))
        x = self.LReLU(self.linear_a3(x))
        s = self.sigmoid(self.linear_s(x))
        x_raw = self.tanh(self.linear_a(x))
        x_normed = self.softmax(x_raw)

        # Sample from Gumbel-softmax variant
        u = torch.rand_like(x_normed)
        action = torch.clip(x_normed * s - 0.01 * torch.log(-torch.log(u)), 0, 1)
        if model_original_out:
            return x, action
        else:
            return action


class openai_critic(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, N, B, action_dim, args):
        super(openai_critic, self).__init__()
        self.N = N
        self.B = B
        self.hidden_dim = args.gnn_hidden_dim
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim

        self.graph_encoders = nn.ModuleList([
            GraphEncoder(
                node_feat_dim=self.node_feat_dim,
                edge_feat_dim=self.edge_feat_dim,
                hidden_dim=self.hidden_dim,
                num_nodes=self.B
            ) for _ in range(self.N)
        ])

        total_input_dim = self.N * (self.B * self.hidden_dim + action_dim)

        self.LReLU = nn.LeakyReLU(0.01)
        self.linear_c1 = nn.Linear(total_input_dim, args.num_units_openai)
        self.linear_c2 = nn.Linear(args.num_units_openai, args.num_units_openai)
        self.linear_c = nn.Linear(args.num_units_openai, 1)

        self.reset_parameters()
        self.train()
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        nn.init.xavier_uniform_(self.linear_c1.weight, gain=gain)
        nn.init.xavier_uniform_(self.linear_c2.weight, gain=gain)
        nn.init.xavier_uniform_(self.linear_c.weight, gain=gain)

    def forward(self, nested_graph, action_input):
        state_embeddings = []
        for i in range(self.N):
            x, edge_index, edge_attr = nested_graph[i]
            z = self.graph_encoders[i](x, edge_index, edge_attr)
            state_embeddings.append(z)

        obs_input = torch.cat(state_embeddings, dim=-1)
        x_cat = self.LReLU(self.linear_c1(torch.cat([obs_input.flatten(), action_input.flatten()])))
        x = self.LReLU(self.linear_c2(x_cat))
        value = self.linear_c(x)
        return value
