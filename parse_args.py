
import argparse
import yaml

def parse_args():
    # default config
    cfg = yaml.load(open('default.yaml', 'r'), Loader=yaml.Loader)
    
    parser = argparse.ArgumentParser(description='CLIP Few Shot')
    
    parser.add_argument('--exp_name',
                        type=str,
                        default="exp")                   
                        
    parser.add_argument('--rand_seed',
                        type=int,
                        default=1)
    parser.add_argument('--torch_rand_seed',
                        type=int,
                        default=1)
    parser.add_argument('--root_path',
                        type=str,
                        default=cfg['root_path'],
                        help='root path of datasets')
    parser.add_argument('--dataset',
                        type=str,
                        default=cfg['dataset'],
                        help='name of dataset')
    parser.add_argument('--shots',
                        type=int,
                        default=cfg['shots'],
                        help='number of shots in each class') 
    parser.add_argument('--train_epoch',
                        type=int,
                        default=cfg['train_epoch'],
                        help='train epochs')
    parser.add_argument('--lr',
                        type=float, default=cfg['lr'], metavar='LR',
                        help='learning rate')
    parser.add_argument('--load_pre_feat',
                        action="store_true",
                        help='load test features or not')
    
    parser.add_argument('--clip_backbone',
                        type=str,
                        default=cfg['clip_backbone'],
                        help='name of clip backbone')
    parser.add_argument('--batch_size',
                        type=int,
                        default=cfg['batch_size'],
                        help='batch size')
    parser.add_argument('--val_batch_size',
                        type=int,
                        default=cfg['val_batch_size'],
                        help='validation batch size')

    parser.add_argument('--num_classes',
                        type=int,
                        default=cfg['num_classes'],
                        help='model classification num') 
    parser.add_argument('--augment_epoch',
                    type=int,
                    default=cfg['augment_epoch'],
                    help='augment epoch')
    parser.add_argument('--load_aux_weight',
                        action="store_true",
                        help='load aux features weight')
    parser.add_argument('--alpha',
                    type=float,
                    default=cfg['alpha'],
                    help='alpha')
    
    parser.add_argument('--lambda_merge',
                    type=float,
                    default=1,
                    help='merge loss ratio'
                    )
    parser.add_argument('--uncent_type',
                    type=str,
                    default='none',
                    help='uncertainty fusion'
                    )
    parser.add_argument('--uncent_power',
                    type=float,
                    default=0.4,
                    help='uncertainty fusion power'
                    )
    return parser