import numpy as np
import torch
from typing import Optional, Tuple
import torch.nn as nn
import torch.nn.functional as F
from net import BFMNet


class Auction(nn.Module):
    def __init__(self, args, oos=False) -> None:
        super().__init__()
        self.n_agents = args.n_agents
        self.device = args.device

        self.menu_size = args.menu_size

        self.epsilon = args.epsilon

        self.mask = (1 - torch.eye(args.n_agents)).to(self.device)

        self.mechanism = BFMNet(args.d_emb, args.n_layer, args.n_head, args.d_hidden, args.d_hidden_idx, args.menu_size - 1, args.n_agents).to(self.device)

    def test_time_forward(self, input_bids: torch.tensor, input_values: torch.tensor, input_budget: torch.tensor):

        bs = input_bids.shape[0]
        bids = input_bids.unsqueeze(1).repeat(1, self.n_agents, 1) * self.mask
        values = input_values.unsqueeze(1).repeat(1, self.n_agents, 1)
        budget_expanded = input_budget.unsqueeze(-1).unsqueeze(-1).repeat(1, self.n_agents, self.n_agents)

        mechanismAlloctionMenu, mechanismPaymentmenu = self.mechanism(bids, values,
                                                                      budget_expanded)  # [bs,n,menu_size-1]
        mechanismAlloctionMenu = torch.ones_like(mechanismAlloctionMenu)
        zero_alloc = torch.zeros((bs, self.n_agents, 1), dtype=mechanismAlloctionMenu.dtype, device=mechanismAlloctionMenu.device)
        zero_payment = torch.zeros((bs, self.n_agents, 1), dtype=mechanismPaymentmenu.dtype, device=mechanismPaymentmenu.device)

        allocation_menu = torch.cat([mechanismAlloctionMenu, zero_alloc], dim=-1)  # bs n ms+1
        payment_menu = torch.cat([mechanismPaymentmenu, zero_payment], dim=-1)  # bs n ms+1

        utility_menu = payment_menu - allocation_menu * input_bids.unsqueeze(-1)

        best_idx = torch.argmax(utility_menu, dim=-1)

        best_idx_expanded = best_idx.unsqueeze(-1)

        allocation = torch.gather(allocation_menu, dim=-1, index=best_idx_expanded).squeeze(-1)
        payment = torch.gather(payment_menu, dim=-1, index=best_idx_expanded).squeeze(-1)
        # =====================================================================

        # valueLoss = -torch.sum(allocation * input_values) / bs  #
        valueLoss = -torch.sum(torch.sqrt(torch.sum(allocation * input_values, dim=-1))) / bs
        B = F.relu(torch.sum(payment, dim=-1) - input_budget)  # (bs,)
        B_Mean = torch.mean(B).to(self.device)
        B_Max = torch.max(B).to(self.device)

        violations = (torch.sum(payment, dim=-1) > input_budget).sum()

        return (B_Mean, B_Max, -valueLoss, torch.sum(payment) / bs, violations)

    def forward(self, input_bids: torch.tensor, input_values: torch.tensor, input_budget: torch.tensor, lamb_bf: torch.tensor, softmax_temp: torch.tensor, cur_sm: float,):
        bs = input_bids.shape[0]
        bids = input_bids.unsqueeze(1).repeat(1, self.n_agents, 1) * self.mask
        values = input_values.unsqueeze(1).repeat(1, self.n_agents, 1)
        budget_expanded = input_budget.unsqueeze(-1).unsqueeze(-1).repeat(1, self.n_agents, self.n_agents)

        mechanismAlloctionMenu, mechanismPaymentmenu = self.mechanism(bids, values, budget_expanded)  # [bs,n,menu_size-1]
        mechanismAlloctionMenu = torch.ones_like(mechanismAlloctionMenu)
        real_utility = mechanismPaymentmenu - mechanismAlloctionMenu * input_bids.unsqueeze(-1)

        zero_alloc = torch.zeros((bs, self.n_agents, 1), dtype=mechanismAlloctionMenu.dtype,
                                 device=mechanismAlloctionMenu.device)
        zero_payment = torch.zeros((bs, self.n_agents, 1), dtype=mechanismPaymentmenu.dtype,
                                   device=mechanismPaymentmenu.device)
        zero_utility = torch.zeros((bs, self.n_agents, 1), dtype=real_utility.dtype, device=real_utility.device)

        if cur_sm > 0:
            top2_utils_real, _ = torch.topk(real_utility, k=2, dim=-1)
            u_second_real = top2_utils_real[..., 1:2]  # [bs, n, 1]

            u_phantom = u_second_real + cur_sm

            zero_phantom = torch.zeros_like(u_phantom)

            _, max_indices_real = torch.max(real_utility, dim=-1, keepdim=True)
            payment_max_real = torch.gather(mechanismPaymentmenu, dim=-1, index=max_indices_real)
            # penalty_payment = input_budget.view(-1, 1, 1).expand(-1, self.n_agents, 1)
            # payment_max_real = penalty_payment

            utility_extended = torch.cat([real_utility, u_phantom, zero_utility], dim=-1)
            allocation_extended = torch.cat([mechanismAlloctionMenu, zero_phantom, zero_alloc], dim=-1)
            payment_extended = torch.cat([mechanismPaymentmenu, payment_max_real, zero_payment], dim=-1)

            softmax_menu = torch.softmax(utility_extended * softmax_temp, dim=-1)

            allocation = torch.sum(allocation_extended * softmax_menu, dim=-1)  # bs n
            payment = torch.sum(payment_extended * softmax_menu, dim=-1)  # bs n
        else:
            utility_menu = torch.cat([real_utility, zero_utility], dim=-1)
            allocation_menu = torch.cat([mechanismAlloctionMenu, zero_alloc], dim=-1)
            payment_menu = torch.cat([mechanismPaymentmenu, zero_payment], dim=-1)

            softmax_menu = torch.softmax(utility_menu * softmax_temp, dim=-1)
            allocation = torch.sum(allocation_menu * softmax_menu, dim=-1)
            payment = torch.sum(payment_menu * softmax_menu, dim=-1)

        # valueLoss = -torch.sum(allocation * input_values) / bs  #
        valueLoss = -torch.sum(torch.sqrt(torch.sum(allocation * input_values, dim=-1) + 1e-6)) / bs
        B = F.relu(torch.sum(payment, dim=-1) - input_budget)  # (bs,)
        B_Mean = torch.mean(B).to(self.device)
        B_Max = torch.max(B).to(self.device)
        lagLoss_B = lamb_bf * B_Mean.to(self.device)
        loss = valueLoss + lagLoss_B

        return (loss, B_Mean, B_Max, -valueLoss, torch.sum(payment) / bs)