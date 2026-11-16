import argparse
import torch
from auction import Auction
from tqdm import tqdm
import logging
import os
import numpy as np
from logger import get_logger

import json

def str2bool(v):
    return v.lower() in ('true', '1')

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

def parse_args():
    parser = argparse.ArgumentParser()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.join(BASE_DIR, 'data/bidder=30')

    parser.add_argument('--data_dir', type=str, default=default_data_dir)
    parser.add_argument('--training_set', type=str, default='train')
    parser.add_argument('--test_set', type=str, default='test')
    parser.add_argument('--final_test_set', type=str, default='final_test')
    parser.add_argument('--n_agents', type=int, default=30)

    parser.add_argument('--d_emb', type=int, default=3)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--d_hidden', type=int, default=256)
    parser.add_argument('--d_hidden_idx', type=int, default=64)
    parser.add_argument('--d_output', type=int, default=1)


    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--train_steps', type=int, default=5000)
    parser.add_argument('--train_sample_num', type=int, default=16384 )
    parser.add_argument('--eval_freq', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--device', type=str, default='cuda:1')

    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--decay_round_one', type=int, default=1500)  #
    parser.add_argument('--one_lr', type=float, default=5e-5)  #
    parser.add_argument('--decay_round_two', type=int, default=2500)  #
    parser.add_argument('--two_lr', type=float, default=1e-5)  #
    parser.add_argument('--decay_round_three', type=int, default=3500)  #
    parser.add_argument('--three_lr', type=float, default=5e-6)  #



    parser.add_argument('--menu_size', type=int, default=15)  #
    parser.add_argument('--softmax_temperature_init', type=int, default=5)

    parser.add_argument('--lamb_bf_init', type=float, default=0.05)
    parser.add_argument('--lamb_bf_update_freq', type=int, default=5)
    parser.add_argument('--epsilon', type=float, default=0.01)

    return parser.parse_args()


def load_data(dir):
    bids = np.load(os.path.join(dir, 'bid.npy')).astype(np.float32)
    values = np.load(os.path.join(dir, 'value.npy')).astype(np.float32)
    budget = np.load(os.path.join(dir, 'budget.npy')).astype(np.float32)
    return (bids, values, budget)


def generate_batch_data_general(batch_size, n_agents, grid_step, device):

    values = torch.rand(batch_size, n_agents, device=device)
    bids = torch.rand(batch_size, n_agents, device=device)
    bids = torch.round(bids / grid_step) * grid_step
    budgets = 0.25 * n_agents + (0.5 * n_agents - 0.25 * n_agents) * torch.rand(batch_size,device=device)

    return bids, values, budgets



if __name__ == "__main__":
    args = parse_args()

    file_path = f"bidder={args.n_agents}"
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    log_path = f"{file_path}/record.log"
    logger = get_logger(log_path)
    logger.info(args)

    torch.manual_seed(args.seed)
    DEVICE = args.device

    test_dir = os.path.join(args.data_dir, args.test_set)
    test_data = load_data(test_dir)

    final_test_dir = os.path.join(args.data_dir, args.final_test_set)
    final_test_data = load_data(final_test_dir)

    model = Auction(args).to(DEVICE)
    cur_softmax_temperature = args.softmax_temperature_init
    cur_lamb_bf = args.lamb_bf_init

    warm_up_init = 1e-8
    warm_up_end = args.lr
    warm_up_anneal_increase = (warm_up_end - warm_up_init) / 50
    optimizer = torch.optim.Adam(model.parameters(), lr=warm_up_init)

    bs = args.batch_size
    num_per_train = int(args.train_sample_num / bs)

    best_valid_value = -float('inf')
    violation_threshold = 0.005

    cur_sm = 0.0

    for i in tqdm(range(args.train_steps)):
        if i == args.train_steps - 1 or ((i >= 1 and i % args.eval_freq == 0)):
            if i == args.train_steps - 1:
                if not os.path.exists("model"):
                    os.makedirs("model")

                model_path = os.path.join("model", str(args.n_agents))
                torch.save(model.state_dict(), model_path)
            with torch.no_grad():
                test_bids = test_data[0]
                test_values = test_data[1]
                test_budget = test_data[2]

                value_sum = 0
                bf_sum = 0
                bf_max = 0
                payment_sum = 0
                violation_sum = 0
                for num in range(int(test_values.shape[0] / bs)):
                    bf_mean_batch, bf_max_batch, batch_value, batch_payment, violations_batch = model.test_time_forward(
                        torch.tensor(test_bids[num * bs:(num + 1) * bs]).to(DEVICE),
                        torch.tensor(test_values[num * bs:(num + 1) * bs]).to(DEVICE),
                        torch.tensor(test_budget[num * bs:(num + 1) * bs]).to(DEVICE),
                    )
                    bf_sum += bf_mean_batch.mean().item() * bs
                    bf_max = max(bf_max, bf_max_batch.item())
                    value_sum += batch_value.item() * bs
                    payment_sum += batch_payment.item() * bs
                    violation_sum += violations_batch.item()

                num_test_samples = test_values.shape[0]

                logger.info(
                    f"Test: epoch={i + 1}, value={(value_sum) / num_test_samples:.6f}, "
                    f"bfp={(bf_sum) / num_test_samples:.6f}, "
                    f"bfp_max={(bf_max) :.6f}, "
                    f"payment={(payment_sum) / num_test_samples:.6f}, "
                    f"violation_rate={(violation_sum) / num_test_samples:.6f}")
                valid_metrics = {
                    "value": float(value_sum / num_test_samples),
                    "bfp": float(bf_sum / num_test_samples),
                    "bfp_max": float(bf_max),
                    "payment": float(payment_sum / num_test_samples),
                    "violation_rate": float(violation_sum / num_test_samples)
                }
                current_value = valid_metrics["value"]
                current_violation = valid_metrics['violation_rate']
                if cur_softmax_temperature >= 10000 and current_value > best_valid_value and current_violation <= violation_threshold:
                    best_valid_value = current_value
                    logging.info(
                        f"Found Better model! Saved -> Epoch: {i + 1}, Value: {current_value:.6f}, Violation: {current_violation:.6f}")

                    model_dir = os.path.join("model", f"bidder={args.n_agents}")
                    if not os.path.exists(model_dir):
                        os.makedirs(model_dir)

                    save_dict = {
                        'state_dict': model.state_dict(),
                        'args': vars(args),
                        'epochs_trained': i + 1,
                        'metrics': valid_metrics
                    }

                    torch.save(save_dict, os.path.join(model_dir, "model_best.pt"))

                    with open(os.path.join(model_dir, "model_info.json"), "w", encoding='utf-8') as f:
                        json.dump({
                            "best_epoch": i + 1,
                            "metrics": valid_metrics,
                            "hyperparameters": vars(args)
                        }, f, indent=4, ensure_ascii=False)

        loss_sum = 0
        value_sum = 0
        bf_sum = 0
        bf_max = 0
        payment_sum = 0

        for num in range(num_per_train):
            batch_bids, batch_values, batch_budget = generate_batch_data_general(bs, args.n_agents, args.epsilon,
                                                                                   DEVICE)

            loss, bf_mean_batch, bf_max_batch, value, payment = model(
                batch_bids,
                batch_values,
                batch_budget,
                cur_lamb_bf,
                torch.tensor(cur_softmax_temperature).to(DEVICE),
                cur_sm
            )
            loss_sum += loss.item() * bs
            bf_sum += bf_mean_batch.mean().item() * bs
            bf_max = max(bf_max, bf_max_batch.item())
            value_sum += value.item() * bs
            payment_sum += payment.item() * bs

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        current_lr = optimizer.param_groups[0]['lr']
        epoch_bf_mean = bf_sum / args.train_sample_num

        if i % 30 == 0:
            cur_softmax_temperature = min(50000.0, cur_softmax_temperature * 1.06)
        if (i + 1) % 15 == 0:
            increment=0.05
            cur_lamb_bf += increment

        if   i % 10 == 0:
            target_sm = args.epsilon / 2.0
            cur_sm = min(target_sm, cur_sm + (target_sm / 200))

        if i % 1 == 0:
            logging.info(f"Train: epoch={i + 1}, loss={loss_sum / args.train_sample_num:.6f}, "
                         f"value={(value_sum) / args.train_sample_num:.6f},"
                         f"bfp={(bf_sum) / args.train_sample_num:.8f}, bfp_max={bf_max:.6f},"
                         f"payment={(payment_sum) / args.train_sample_num:.6f},"
                         f"lamb_bfp={cur_lamb_bf:.4f} ,"
                         f"softmax_temperature={cur_softmax_temperature:.2f},"
                         f"current_learning_rate={current_lr:.6f} ,"
                         f"sm={cur_sm:.6f}")

        if i <= 50:  # warm up
            for p in optimizer.param_groups:
                p['lr'] += warm_up_anneal_increase

        if i == args.decay_round_one:
            for p in optimizer.param_groups:
                p['lr'] = args.one_lr

        if i == args.decay_round_two:
            for p in optimizer.param_groups:
                p['lr'] = args.two_lr

        if i == args.decay_round_three:
            for p in optimizer.param_groups:
                p['lr'] = args.three_lr

    logger.info("------------Final test------------")
    with torch.no_grad():
        final_test_bids = final_test_data[0]
        final_test_values = final_test_data[1]
        final_test_budget = final_test_data[2]

        final_value_sum = 0
        final_bf_sum = 0
        final_bf_max = 0
        final_payment_sum = 0
        final_violation_sum = 0

        num_final_batches = int(final_test_values.shape[0] / bs)

        for num in range(num_final_batches):
            bf_mean_batch, bf_max_batch, value, payment, violations_batch = model.test_time_forward(
                torch.tensor(final_test_bids[num * bs:(num + 1) * bs]).to(DEVICE),
                torch.tensor(final_test_values[num * bs:(num + 1) * bs]).to(DEVICE),
                torch.tensor(final_test_budget[num * bs:(num + 1) * bs]).to(DEVICE),
            )

            final_bf_sum += bf_mean_batch.mean().item() * bs
            final_bf_max = max(final_bf_max, bf_max_batch.item())
            final_value_sum += value.item() * bs
            final_payment_sum += payment.item() * bs
            final_violation_sum += violations_batch.item()

        num_final_test_samples = final_test_values.shape[0]

        logger.info(
            f"value={(final_value_sum) / num_final_test_samples:.6f}, "
            f"bfp={(final_bf_sum) / num_final_test_samples:.6f}, "
            f"bfp_max={(final_bf_max) :.6f}, "
            f"payment={(final_payment_sum) / num_final_test_samples:.6f}, "
            f"violation_rate={(final_violation_sum) / num_final_test_samples:.6f}")