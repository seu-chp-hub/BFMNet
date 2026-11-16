
import torch
import torch.nn as nn
import torch.nn.functional as F


class BFMNet(nn.Module):
    def __init__(self, d_emb, n_layer, n_head, d_hidden,d_hidden_idx, menu_size,n_agents):
        super(BFMNet, self).__init__()
        self.menu_size = menu_size

        d_output=2*menu_size

        self.pre_net = nn.Sequential(
            nn.Linear(d_emb, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden)
        )
        self.d_hidden_idx = d_hidden_idx
        self.agent_embedding = nn.Embedding(n_agents, self.d_hidden_idx)
        nn.init.trunc_normal_(self.agent_embedding.weight, std=0.02)
        d_model = d_hidden + self.d_hidden_idx

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_hidden,
            dropout=0.0,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        self.fc_out = nn.Linear(d_model, d_output)

        self.alloc_output_dim =  nn.Sequential(
            nn.Linear(menu_size, menu_size),
            nn.ReLU(),
            nn.Linear(menu_size, menu_size)
        )

        self.payment_output_dim =  nn.Sequential(
            nn.Linear(menu_size, menu_size),
            nn.ReLU(),
            nn.Linear(menu_size, menu_size)
        )


    def forward(self, batch_bids, batch_values, batch_budget):
        """
        batch_bids: [bs, n, n]
        batch_values: [bs, n, n]
        batch_budget: [bs, n, n]
        """
        bs, n_bidder = batch_bids.shape[0], batch_bids.shape[1]

        x1 = batch_bids.unsqueeze(-1)  # [bs, n, n, 1]
        x2 = batch_values.unsqueeze(-1)
        x3 = batch_budget.unsqueeze(-1)

        x_raw = torch.cat([x1, x2, x3], dim=-1) # [bs, n, n, 3]
        x_feat = self.pre_net(x_raw) # [bs, n, n, d_hidden]

        # [bs, n, d_hidden_id]
        agent_ids = torch.arange(n_bidder, device=x_raw.device).view(1, 1, n_bidder).expand(bs, n_bidder, n_bidder)
        x_id = self.agent_embedding(agent_ids)  # [bs, n, n, d_hidden_idx]

        # max_vocab_size = self.agent_embedding.num_embeddings
        #
        # base_ids = torch.arange(n_bidder, device=x_raw.device) % max_vocab_size
        #
        # agent_ids = base_ids.view(1, 1, n_bidder).expand(bs, n_bidder, n_bidder)
        # x_id = self.agent_embedding(agent_ids)  # [bs, n, n, d_hidden_idx]

        x_combined = torch.cat([x_feat, x_id], dim=-1)  # [bs, n, n, d_model]

        d_model = x_combined.shape[-1]
        x_combined_flat = x_combined.reshape(bs * n_bidder, n_bidder, d_model)
        x_trans_flat = self.transformer(x_combined_flat)

        x_trans = x_trans_flat.reshape(bs, n_bidder, n_bidder, d_model)
        mechanism_out = self.fc_out(x_trans)

        alloc_out= self.alloc_output_dim(mechanism_out[:,:,:,:self.menu_size]) # bs, n, n, menu_size
        payment_out = self.payment_output_dim(mechanism_out[:,:,:,self.menu_size:])

        allocation = torch.sigmoid(torch.mean(alloc_out, dim=2))  #  [bs, n, menu_size]

        payment = F.softplus(torch.mean(payment_out,dim=2))


        return allocation, payment


