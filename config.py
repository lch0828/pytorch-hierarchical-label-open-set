import argparse

def print_config(config, logger=None):
    config = vars(config)
    info = "configs:\n"
    for k, v in config.items():
        info += "\t{} : {}\n".format(k, str(v))
    if not logger:
        print("\n" + info + "\n")
    else:
        logger.info("\n" + info + "\n")


def conf():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=100, type=int) 
    parser.add_argument('--batch', default=32, type=int)

    parser.add_argument('--train_div', default='../train_shuffle')
    parser.add_argument('--test_div', default='../test_shuffle')

    parser.add_argument('--in_dim', default=768, type=int)
    parser.add_argument('--global_weight_dim', default=1536, type=int)
    parser.add_argument('--transition_weight_dim', nargs='+', default=[128, 512])
    parser.add_argument('--total_classes_at_level', nargs='+', default=[4, 88])
    parser.add_argument('--total_levels', default=2, type=int)
    parser.add_argument('--B', default=0.5, type=float)

    parser.add_argument('--encoder', default='vit_b_16', choices=['vit_b_16', 'vit_b_32', 'cnn'])
    parser.add_argument('--test_model', nargs='?')
    parser.add_argument('--deep_residual', action='store_true')

    args = parser.parse_args()
    
    return args
