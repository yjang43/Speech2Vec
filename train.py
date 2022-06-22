import os
import torch
import torch.nn as nn
import torch.optim as optim

from argparse import ArgumentParser
from model import *
from dataset import *



def set_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--data_dir', 
        default='data/', 
        type=str,
        help='Directory that saves .npy data files'
    )
    parser.add_argument(
        '--word_dir', 
        default='data_mapping/', 
        type=str,
        help='Directory that saves .txt word mapping files'
    )
    parser.add_argument(
        '--window_sz', 
        default=3, 
        type=int,
        help='Window size for language modeling'
    )
    parser.add_argument(
        '--mfcc_dim', 
        default=13, 
        type=int,
        help='The number of mfcc feature extract'
    )
    parser.add_argument(
        '--embedding_dim', 
        default=50, 
        type=int,
        help='The size of word embedding'
    )
    parser.add_argument(
        '--lr', 
        default=1e-3, 
        type=float,
        help='Learning rate'
    )
    parser.add_argument(
        '--epochs', 
        default=10, 
        type=int,
        help='The number of epochs'
    )
    parser.add_argument(
        '--device', 
        default='cpu', 
        type=str,
        help='Device to train on cpu/cuda'
    )
    parser.add_argument(
        '--print_itr',
        default=10, 
        type=int,
        help='The number of iteration to log training process'
    )
    parser.add_argument(
        '--ckpt_dir', 
        default='ckpt/', 
        type=str,
        help='Directory to save checkpoints'
    )
    parser.add_argument(
        '--ckpt_n', 
        default='', 
        type=str,
        help='The name of checkpoint to continue training from, '\
             'do not load if left empty'
    )
    args = parser.parse_args()
    return args


def save_ckpt(ckpt_dir, ckpt_n, keep_topk=5, **kwargs):
    fs = sorted([f for f in os.listdir(ckpt_dir) if f[-3: ] == '.pt'], 
                key=lambda f: int(f.split('.')[0]))
    
    if keep_topk <= len(fs):     # trick to delete ckpt
        os.rename(os.path.join(ckpt_dir, fs[0]), os.path.join(ckpt_dir, ckpt_n))
    
    # make sure state_dict is in cpu()
    for p in kwargs['state_dict']:
        kwargs['state_dict'][p].cpu()
    torch.save(kwargs, os.path.join(ckpt_dir, ckpt_n))

    
    
if __name__ == '__main__':
    args = set_args()
    
    # prepare checkpoint
    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt = torch.load(os.path.join(args.ckpt_dir, args.ckpt_n)) if args.ckpt_n else None

    # load model and prepare data
    model = Speech2Vec(
        input_dim=args.mfcc_dim, 
        hidden_dim=args.embedding_dim, 
        window_sz=args.window_sz
    )
    model.to(args.device)

    # dataset = LibriSpeechDataset(data_dir=args.data_dir, word_dir=args.word_dir, window_sz=args.window_sz)
    dataset = LibriSpeechDatasetFast(
        data_dir=args.data_dir, 
        word_dir=args.word_dir, 
        window_sz=args.window_sz
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_sz, 
        num_workers=2,    # comment if dataset suffers from IO overhead
        shuffle=True, 
        collate_fn=dataset.pad_collate)


    if ckpt is not None:
        # TODO: validate on args
        # model load from ckpt
        model.cpu()
        model.load_state_dict(ckpt['state_dict'])
        model.to(args.device)

        # status variables load from ckpt
        from_epoch = ckpt['epoch'] + 1
        itr = ckpt['itr']
        losses = ckpt['losses']

    else:
        # initialize status variables
        from_epoch = 0
        itr = 0
        losses = []
        

    criterion = nn.MSELoss()
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr)

    train_progbar = tqdm(range((args.epochs - from_epoch) * len(dataloader)))

    for epoch in range(from_epoch, args.epochs):
        model.train()
        for batch_idx, batch in enumerate(dataloader):
            src, tgts, _, _ = batch
            src = [s.to(args.device) for s in src]
            tgts = [[t.to(args.device) for t in tgt] for tgt in tgts]

            optimizer.zero_grad()

            # pred is padded sequence
            preds = model(src, tgts)

            for k in range(2 * args.window_sz):
                if k == 0:
                    loss = torch.zeros(1)
                # tgt likewise need to be padded sequence.
                tgt = nn.utils.rnn.pad_sequence(tgts[k])[1: ]
                loss += criterion(preds[k], tgt)

            loss.backward()
            optimizer.step()
            # TODO: Clipping to prevent gradient explosion
            nn.utils.clip_grad.clip_grad_norm(model.parameters(), max_norm=5.0)

            if (itr + 1) % args.print_itr == 0:
                loss_item = loss.item()
                losses.append(loss_item)
                train_progbar.set_description((f"Epoch: {round(epoch + batch_idx / len(dataloader), 3)} | " \
                                              f"Loss: {round(loss_item, 3)}"))
                train_progbar.refresh()
                
            train_progbar.update()
            itr += 1

        # TODO: evaluation code
        # TODO: logging information about evaluation
        # evaluation code could be added here
        model.eval()
        
        # save model after epoch
        save_ckpt(
            args.ckpt_dir, 
            f'{epoch + 1}.pt', 
            state_dict=model.state_dict(), 
            itr=itr, 
            epoch=epoch, 
            losses=losses
        )
