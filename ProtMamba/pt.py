import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from data import FastaDataset
from torch.utils.data import DataLoader, DistributedSampler
from ProtMamba import protmamba
from loss_fn import masked_cross_entropy_loss
from tqdm import tqdm
import os
import argparse

def train_batch(model, loss_fn, optimizer, data, device, epoch, total_epochs):
    model.train()
    optimizer.zero_grad()
    m_x, cls_x = model(data['masked_seq'].to('cuda'))
    loss = masked_cross_entropy_loss(m_x, data['raw_seq'].to('cuda'), data['mask_pos'].to('cuda'))
    loss.backward()
    optimizer.step()
    return loss

def train_epoch(rank, model, loss_fn, optimizer,
                tr_data_loader, device, log_file,
                output_dir, epoch, total_epochs,
                lr, warmup, model_config):
    if rank == 0:
        pbar = tqdm(total=len(tr_data_loader), desc=f"Epoch {epoch}/{total_epochs}", unit='batch')
    total_loss = 0.0
    for batch_idx, batch in enumerate(tr_data_loader):
        lr_this_step = lr * warmup_linear(epoch / total_epochs, warmup=warmup)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_this_step
        loss = train_batch(model, loss_fn, optimizer, batch, device, epoch, total_epochs)
        total_loss += loss
        reduced_loss = torch.tensor([total_loss], device=device)
        dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
        reduced_loss = reduced_loss.item()
        avg_loss = reduced_loss / ((batch_idx + 1) * dist.get_world_size())
        if rank == 0:
            pbar.update(1)
            pbar.set_postfix({'avg_loss': avg_loss, 'lr': param_group['lr']})
            with open(log_file, 'a') as f:
                f.write(f"Epoch {epoch}/{total_epochs} Batch {batch_idx+1}/{len(tr_data_loader)} Loss {avg_loss} lr {param_group['lr']}\n")

    if rank == 0:
        pbar.close()
        with open(log_file, 'a') as f:
            f.write(f"Epoch Loss: {avg_loss}\n")
        model_output_path = f"{output_dir}/model_epoch_{epoch}.pt"
        torch.save({
            'config': model_config,
            'model_state_dict': model.state_dict(),
        }, model_output_path)
    if rank == 0:
        print(f"Epoch {epoch} - Average Loss: {avg_loss}")

def warmup_linear(x, warmup=0.01):
    if warmup == 0.0:
        return 1.0
    else:
        if x == 0:
            return 0.01
        elif x != 0 and x < warmup:
            return 1.0 - x
        else:
            return 1.0

def train(rank, seq_len, dict_path, mask_rate, ckpt, model_config, num_epochs, data_path, output_path, bsz, lr, warmup, num_workers):
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(rank)

    if ckpt is not None:
        checkpoint = torch.load(ckpt, map_location=device)
        model_config = checkpoint['config']
        model = protmamba(**model_config).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model = protmamba(**model_config).to(device)

    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    tr_dataset = FastaDataset(data_path, seq_len, dict_path, mask_rate)
    tr_sampler = DistributedSampler(tr_dataset, num_replicas=dist.get_world_size(), rank=rank)
    tr_data_loader = DataLoader(tr_dataset, batch_size=bsz, shuffle=False, num_workers=num_workers, sampler=tr_sampler)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    if rank == 0:
        os.makedirs(output_path, exist_ok=True)
    log_file = os.path.join(output_path, 'pt.log') if rank == 0 else None
    for epoch in range(num_epochs):
        if rank == 0:
            print(f'Epoch {epoch}')
        train_epoch(rank, model, loss_fn, optimizer, tr_data_loader,
                    device, log_file, output_path, epoch, num_epochs,
                    lr, warmup, model_config)
    if rank == 0:
        print("Training finished!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', default=0, type=int, help='Local rank. Necessary for using the torch.distributed.launch utility.')
    args = parser.parse_args()
    torch.distributed.init_process_group(backend='nccl')
    rank = args.local_rank
    data_path = './uniref100.fasta'
    output_path = './out/'
    seq_len = 2048
    dict_path = './vocab.txt'
    mask_rate = 0.15
    ckpt = None
    num_epochs = 500
    bsz = 2
    lr = 1e-3
    warmup = 0.00 # 0.02
    num_workers = 8
    ckpt = None
    model_config = {'seqlen':seq_len, 
                'vocab':28,
                'n_layers':12, 
                'd_model':768, 
                'expand_factor':2,
                'd_state':16, 
                'pscan':True, 
                'num_cls':2, 
                'cls_mlp_droprate':0.1, 
                'dec_mlp_droprate':0.1}
    train(rank, seq_len, dict_path, mask_rate, ckpt, model_config, num_epochs, data_path, output_path, bsz, lr, warmup, num_workers)