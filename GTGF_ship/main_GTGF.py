import argparse
import os
import time
from frame.processor import processor
import warnings
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser( description='GTGF')

parser.add_argument('--data_root', default='./AIS_process/AIS_processed.cpkl', type=str)
parser.add_argument('--save_dir', default='./output/', help='Directory for saving caches and models.')

parser.add_argument('--train_model', default='GTGF', help='[GTGF, LSTM]')
parser.add_argument('--data_rate', default=[8,0,2], type=list) # train, val, test
parser.add_argument('--load_model', default='best', type=str, help="load pretrained model for test or training")

parser.add_argument('--seq_length', default=40, type=int)
parser.add_argument('--obs_length', default=16, type=int)
parser.add_argument('--pred_length', default=24, type=int)
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--learning_rate', default=0.0005, type=float)
parser.add_argument('--clip', default=1, type=int)
parser.add_argument('--dropout', default=0.1, type=float)
# model_TS_transformer
parser.add_argument('--emsize', default=256, type=int, help='embedding dimension')
parser.add_argument('--nhid', default=128, type=int, help='the dimension of the feedforward network model in TransformerEncoder')
parser.add_argument('--nlayers', default=1, type=int, help='the number of nn.TransformerEncoderLayer in nn.TransformerEncoder')
parser.add_argument('--nhead', default=8, type=int, help='the number of heads in the multihead-attention models')
parser.add_argument('--K', default=20, type=int, help='number of generating trajectory')
parser.add_argument('--device', default='cuda:2', type=str)
# Glow
parser.add_argument('--feats_in', default=5, type=int)
parser.add_argument('--feats_hidden', default=256, type=int)
parser.add_argument('--feats_out', default=2, type=int)
parser.add_argument('--flows', default=16, type=int, help= 'number of flows in all blocks')
parser.add_argument('--blocks', default=4, type=int)

args = parser.parse_args()

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

trainer = processor(args)

ts = time.time()
trainer.train()
print('train_time: {}-----------------'.format(time.time()-ts))

trainer.test()
trainer.pred()