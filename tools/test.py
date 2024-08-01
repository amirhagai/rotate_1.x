# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

from mmdet.utils import register_all_modules as register_all_modules_mmdet
from mmengine.config import Config, DictAction
from mmengine.evaluator import DumpResults
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmengine.runner.amp import autocast
from mmrotate.utils import register_all_modules


# TODO: support fuse_conv_bn and format_only
def parse_args():
    parser = argparse.ArgumentParser(description='Test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--out',
        type=str,
        help='dump predictions to a pickle file for offline evaluation')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def trigger_visualization_hook(cfg, args):
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        # Turn on visualization
        visualization_hook['draw'] = True
        if args.show:
            visualization_hook['show'] = True
            visualization_hook['wait_time'] = args.wait_time
        if args.show_dir:
            visualization_hook['test_out_dir'] = args.show_dir
    else:
        raise RuntimeError(
            'VisualizationHook must be included in default_hooks.'
            'refer to usage '
            '"visualization=dict(type=\'VisualizationHook\')"')

    return cfg


def main():
    args = parse_args()

    # register all modules in mmdet into the registries
    # do not init the default scope here because it will be init in the runner
    register_all_modules_mmdet(init_default_scope=False)
    register_all_modules(init_default_scope=False)

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # add `DumpResults` dummy metric
    if args.out is not None:
        assert args.out.endswith(('.pkl', '.pickle')), \
            'The dump file must be a pkl file.'
        runner.test_evaluator.metrics.append(
            DumpResults(out_file_path=args.out))

    # start testing
    runner.test()


def get_outputs():
    
    
    args = parse_args()

    # register all modules in mmdet into the registries
    # do not init the default scope here because it will be init in the runner
    register_all_modules_mmdet(init_default_scope=False)
    register_all_modules(init_default_scope=False)

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # add `DumpResults` dummy metric
    if args.out is not None:
        assert args.out.endswith(('.pkl', '.pickle')), \
            'The dump file must be a pkl file.'
        runner.test_evaluator.metrics.append(
            DumpResults(out_file_path=args.out))

    # start testing
    test_loop = runner.build_test_loop(runner._test_loop)
    
    
    test_loop.runner.model.eval()
    outs = []
    for idx, data_batch in enumerate(test_loop.dataloader):
        # test_loop.run_iter(idx, data_batch)
        with autocast(enabled=test_loop.fp16):
            outputs = test_loop.runner.model.test_step(data_batch)
        outs.append(outputs)
    return outs


if __name__ == '__main__':
    # get_outputs_over_batch()
    # from torch import tensor
    # import matplotlib.pyplot as plt
    # import numpy as np
    
    # flat_container_scores = np.array([0.6306, 0.5623, 0.5602, 0.5407, 0.5163, 0.4825, 0.4669, 0.4169, 0.4129,
    #     0.4127, 0.3918, 0.3771, 0.3469, 0.3269, 0.3221, 0.3072, 0.3063, 0.2941,
    #     0.2913, 0.2880, 0.2854, 0.2802, 0.2608, 0.2432, 0.2429, 0.2294, 0.2241,
    #     0.2239, 0.2210, 0.2191, 0.2186, 0.2149, 0.2125, 0.2072, 0.2018, 0.2010,
    #     0.1935, 0.1923, 0.1808, 0.1788, 0.1729, 0.1629, 0.1458, 0.1446, 0.1376,
    #     0.1338, 0.1307, 0.1034, 0.1004, 0.0945, 0.0899, 0.0867, 0.0866, 0.0796,
    #     0.0761, 0.0759, 0.0747, 0.0742, 0.0709, 0.0701, 0.0681, 0.0642, 0.0630,
    #     0.0616, 0.0612, 0.0597, 0.0574, 0.0571, 0.0558, 0.0551, 0.0528, 0.0503,
    #     0.0502])
    
    # flat_container_labels = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 5, 5, 5, 5, 5, 4, 5, 4, 5, 5, 4, 4, 4, 5,
    #     5, 5, 5, 5, 5, 4, 4, 5, 5, 5, 4, 5, 5, 4, 4, 5, 5, 5, 4, 5, 4, 4, 5, 4,
    #     4, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 6, 5, 5, 5, 0, 4, 5, 5, 4, 4, 4,
    #     4])
    
    
    # truck_scores = np.array([0.6125, 0.5143, 0.5010, 0.4953, 0.4821, 0.3579, 0.3513, 0.3404, 0.3264,
    #     0.3226, 0.3216, 0.3085, 0.3071, 0.2826, 0.2518, 0.2312, 0.2132, 0.2073,
    #     0.2035, 0.1913, 0.1903, 0.1892, 0.1811, 0.1690, 0.1688, 0.1682, 0.1632,
    #     0.1572, 0.1511, 0.1500, 0.1414, 0.1412, 0.1406, 0.1386, 0.1360, 0.1280,
    #     0.1210, 0.1141, 0.1136, 0.1132, 0.1132, 0.1103, 0.1076, 0.1074, 0.0993,
    #     0.0987, 0.0961, 0.0941, 0.0935, 0.0926, 0.0896, 0.0856, 0.0845, 0.0836,
    #     0.0791, 0.0766, 0.0742, 0.0724, 0.0720, 0.0705, 0.0693, 0.0691, 0.0689,
    #     0.0678, 0.0676, 0.0673, 0.0582, 0.0577, 0.0571, 0.0544, 0.0542, 0.0539,
    #     0.0533, 0.0531, 0.0520, 0.0508, 0.0507])
    
    # truck_labels = np.array([ 5,  5,  5,  5,  5,  5,  5,  5,  4,  5,  5,  5,  5,  5,  5,  5,  5,  4,
    #      5,  5,  4,  5,  5,  5,  5,  4,  5,  5,  5,  5,  4, 14,  5,  5,  4,  5,
    #      4,  5,  5,  5,  7,  4,  5,  4,  7,  5, 14,  4,  5,  4,  4, 14,  4,  4,
    #      5,  4,  4,  4,  5,  7,  4,  4,  4,  5,  4,  5,  4,  4,  4,  4, 14,  0,
    #      4,  4, 14,  5,  4])
    
    # normal_container = np.array([0.6406, 0.4936, 0.4769, 0.4594, 0.4541, 0.4510, 0.4443, 0.4242, 0.3999,
    #     0.3864, 0.3760, 0.3701, 0.3350, 0.3012, 0.2864, 0.2829, 0.2813, 0.2757,
    #     0.2645, 0.2590, 0.2579, 0.2466, 0.2440, 0.2415, 0.2187, 0.2151, 0.2144,
    #     0.2026, 0.2022, 0.1957, 0.1918, 0.1897, 0.1883, 0.1818, 0.1758, 0.1698,
    #     0.1677, 0.1665, 0.1656, 0.1627, 0.1583, 0.1488, 0.1466, 0.1436, 0.1407,
    #     0.1355, 0.1174, 0.1016, 0.0964, 0.0950, 0.0949, 0.0922, 0.0915, 0.0894,
    #     0.0881, 0.0861, 0.0807, 0.0802, 0.0795, 0.0782, 0.0774, 0.0767, 0.0747,
    #     0.0746, 0.0708, 0.0698, 0.0693, 0.0684, 0.0677, 0.0637, 0.0622, 0.0602,
    #     0.0565, 0.0565, 0.0560, 0.0554, 0.0542, 0.0529, 0.0526, 0.0523, 0.0519,
    #     0.0518, 0.0514, 0.0508, 0.0502])
    
    # normal_container_labels = np.array([5, 5, 5, 5, 5, 5, 5, 5, 4, 5, 5, 5, 5, 4, 4, 5, 5, 5, 5, 5, 5, 5, 4, 5,
    #         5, 5, 5, 5, 5, 4, 5, 4, 5, 5, 5, 5, 4, 4, 5, 5, 5, 5, 4, 5, 4, 5, 5, 4,
    #         5, 4, 4, 4, 5, 4, 4, 4, 4, 4, 4, 4, 5, 4, 4, 4, 5, 5, 5, 4, 5, 5, 4, 5,
    #         5, 6, 5, 4, 6, 4, 6, 5, 4, 6, 4, 4, 5])


    # # Sample tensors (replace these with your actual tensors)
    # tensor1 = flat_container_scores[flat_container_labels == 5]
    # tensor2 = normal_container[normal_container_labels == 5]
    # tensor3 = truck_scores[truck_labels == 5]
    # # Set up the figure and axes for three subplots
    # fig, axs = plt.subplots(1, 3, figsize=(18, 10))

    # # Histogram for tensor1
    # axs[0].hist(tensor1, bins=20, color='skyblue', edgecolor='black')
    # axs[0].set_title('scores for flat texture container scores', fontsize=15)
    # axs[0].set_xlabel('Value', fontsize=12)
    # axs[0].set_ylabel('Frequency', fontsize=12)
    # axs[0].grid(axis='y', alpha=0.75)
    # axs[0].set_xticks(np.arange(0, 1.1, 0.1))
    # axs[0].tick_params(axis='x', labelsize=10)
    # axs[0].tick_params(axis='y', labelsize=10)
    # axs[0].set_ylim(0, 10)

    # # Histogram for tensor2
    # axs[1].hist(tensor2, bins=20, color='lightgreen', edgecolor='black')
    # axs[1].set_title('Histogram normal texture rendering container', fontsize=15)
    # axs[1].set_xlabel('Value', fontsize=12)
    # axs[1].set_ylabel('Frequency', fontsize=12)
    # axs[1].grid(axis='y', alpha=0.75)
    # axs[1].set_xticks(np.arange(0, 1.1, 0.1))
    # axs[1].tick_params(axis='x', labelsize=10)
    # axs[1].tick_params(axis='y', labelsize=10)
    # axs[1].set_ylim(0, 10)

    # # Histogram for tensor3
    # axs[2].hist(tensor3, bins=20, color='salmon', edgecolor='black')
    # axs[2].set_title('Histogram of trucks scores', fontsize=15)
    # axs[2].set_xlabel('Value', fontsize=12)
    # axs[2].set_ylabel('Frequency', fontsize=12)
    # axs[2].grid(axis='y', alpha=0.75)
    # axs[2].set_xticks(np.arange(0, 1.1, 0.1))
    # axs[2].tick_params(axis='x', labelsize=10)
    # axs[2].tick_params(axis='y', labelsize=10)
    # axs[2].set_ylim(0, 10)

    # # Adjust layout to prevent overlap
    # plt.tight_layout()
    # plt.savefig('/app/data/hist.png')
    # # Show plot
    # plt.show()
    # plt.close()
    
    
    # tensor1 = tensor1[tensor1 > 0.3]
    # tensor2 = tensor2[tensor2 > 0.3]
    # tensor3 = tensor3[tensor3 > 0.3]
    
    # fig, axs = plt.subplots(1, 3, figsize=(30, 10))

    # # Histogram for tensor1
    # axs[0].hist(tensor1, bins=20, color='skyblue', edgecolor='black')
    # axs[0].set_title('scores for flat texture container scores', fontsize=15)
    # axs[0].set_xlabel('Value', fontsize=12)
    # axs[0].set_ylabel('Frequency', fontsize=12)
    # axs[0].grid(axis='y', alpha=0.75)
    # axs[0].set_xticks(np.arange(0, 1.1, 0.1))
    # axs[0].tick_params(axis='x', labelsize=10)
    # axs[0].tick_params(axis='y', labelsize=10)
    # axs[0].set_ylim(0, 3)

    # # Histogram for tensor2
    # axs[1].hist(tensor2, bins=20, color='lightgreen', edgecolor='black')
    # axs[1].set_title('Histogram normal texture rendering container', fontsize=15)
    # axs[1].set_xlabel('Value', fontsize=12)
    # axs[1].set_ylabel('Frequency', fontsize=12)
    # axs[1].grid(axis='y', alpha=0.75)
    # axs[1].set_xticks(np.arange(0, 1.1, 0.1))
    # axs[1].tick_params(axis='x', labelsize=10)
    # axs[1].tick_params(axis='y', labelsize=10)
    # axs[1].set_ylim(0, 3)

    # # Histogram for tensor3
    # axs[2].hist(tensor3, bins=20, color='salmon', edgecolor='black')
    # axs[2].set_title('Histogram of trucks scores', fontsize=15)
    # axs[2].set_xlabel('Value', fontsize=12)
    # axs[2].set_ylabel('Frequency', fontsize=12)
    # axs[2].grid(axis='y', alpha=0.75)
    # axs[2].set_xticks(np.arange(0, 1.1, 0.1))
    # axs[2].tick_params(axis='x', labelsize=10)
    # axs[2].tick_params(axis='y', labelsize=10)
    # axs[2].set_ylim(0, 3)

    # # Adjust layout to prevent overlap
    # plt.tight_layout()
    # plt.savefig('/app/data/hist>0.3.png')
    # # Show plot
    # plt.show()


    
    main()
