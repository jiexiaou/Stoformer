import os
import torch


class Options():
    """docstring for Options"""

    def __init__(self):
        pass

    def init(self, parser):
        parser.add_argument('--batch_size', type=int, default=8, help='batch size')
        parser.add_argument('--nepoch', type=int, default=250, help='training epochs')
        parser.add_argument('--train_workers', type=int, default=4, help='train_dataloader workers')
        parser.add_argument('--eval_workers', type=int, default=1, help='eval_dataloader workers')
        parser.add_argument('--optimizer', type=str, default='adam', help='optimizer for training')
        parser.add_argument('--lr_initial', type=float, default=0.0002, help='initial learning rate')
        parser.add_argument('--LR_MIN', type=float, default=1e-6)
        parser.add_argument('--thre', type=int, default=50)
        parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
        parser.add_argument('--gpu', type=str, default='0,1', help='GPUs')
        parser.add_argument('--arch', type=str, default='Stoformer', help='archtechture')

        parser.add_argument('--save_dir', type=str, default='', help='save dir')
        parser.add_argument('--save_images', action='store_true', default=False)
        parser.add_argument('--env', type=str, default='_', help='env')
        parser.add_argument('--checkpoint', type=int, default=50, help='checkpoint')

        parser.add_argument('--norm_layer', type=str, default='nn.LayerNorm', help='normalize layer in transformer')
        parser.add_argument('--embed_dim', type=int, default=32, help='dim of emdeding features')
        parser.add_argument('--win_size', type=int, default=8, help='window size of self-attention')
        parser.add_argument('--token_projection', type=str, default='linear', help='linear/conv token projection')
        parser.add_argument('--token_mlp', type=str, default='leff', help='ffn/leff token mlp')
        parser.add_argument('--att_se', action='store_true', default=False, help='se after sa')

        parser.add_argument('--noiselevel', type=float, default=50)
        parser.add_argument('--use_grad_clip', action='store_true', default=False)

        # args for training
        parser.add_argument('--train_ps', type=int, default=128, help='patch size of training sample')
        parser.add_argument('--train_dir', type=str, default='', help='dir of train data')
        parser.add_argument('--val_dir', type=str, default='', help='dir of train data')
        parser.add_argument('--random_start', type=int, default=0, help='epochs for random shift')

        # args for testing
        parser.add_argument('--weights', type=str, default='', help='Path to trained weights')
        parser.add_argument('--test_workers', type=int, default=1, help='number of test works')
        parser.add_argument('--input_dir', type=str, default='', help='Directory of validation images')
        parser.add_argument('--result_dir', type=str, default='', help='Directory for results')
        parser.add_argument('--crop_size', type=int, default=256, help='crop size for testing')
        parser.add_argument('--overlap_size', type=int, default=30, help='overlap size for testing')
        return parser
