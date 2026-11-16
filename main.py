import warnings
warnings.filterwarnings("ignore")
import argparse
from utils import *

from train_graph_diffusion import train_graph_diffusion
from train_graph_latent_diffusion import train_graph_latent_diffusion
from test_graph_diffusion import test_graph_diffusion
from test_graph_latent_diffusion import test_graph_latent_diffusion

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/graph_diffusion.yaml')

    parser.add_argument('--device', type=str, default=None, help='override config device, e.g. cuda:1')
    parser.add_argument('--exp_name', type=str, default=None, help='sub-folder for this run (ablations)')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--pipeline', type=str, default=None, help='override pipeline (train_/test_graph_latent_diffusion)')
    parser.add_argument('--ckpt', type=str, default=None, help='override test_config.ckpt')
    parser.add_argument('--set', nargs='*', default=[],
                        help='generic overrides: section.key=value, e.g. model_config.use_prior=false')

    args = parser.parse_args()
    config = load_config(args.config)
    if args.device is not None:
        config['device'] = args.device
    if args.exp_name is not None:
        config['exp_name'] = args.exp_name
    if args.seed is not None:
        config['seed'] = args.seed
    if args.pipeline is not None:
        config['pipeline'] = args.pipeline
    if args.ckpt is not None:
        config.setdefault('test_config', {})['ckpt'] = args.ckpt
    import yaml
    for kv in args.set:
        k, v = kv.split('=', 1)
        node = config
        keys = k.split('.')
        for kk in keys[:-1]:
            node = node.setdefault(kk, {})
        node[keys[-1]] = yaml.safe_load(v)
    pipeline = config['pipeline']

    if pipeline == 'train_graph_latent_diffusion':
        train_graph_latent_diffusion(config)
    elif pipeline == 'train_graph_diffusion':
        train_graph_diffusion(config)
    elif pipeline == 'test_graph_diffusion':
        test_graph_diffusion(config)
    elif pipeline == 'test_graph_latent_diffusion':
        test_graph_latent_diffusion(config)
    else:
        raise ValueError(f'pipeline {pipeline} not recognized')

