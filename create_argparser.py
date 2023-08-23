import argparse
from configargparse import Parser

def parse_float_list(input_string):
    try:
        return [float(item) for item in input_string.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid float list format. Please provide a comma-separated list of numbers.")

def int_or_list(s):
    breakpoint()
    try:
        # Try to parse as a single int
        return int(s)
    except ValueError:
        # If parsing as a single int fails, split by commas and parse as a list of ints
        return [int(x) for x in s.split(',')]


def get_parser_args():
    parser = Parser(
        args_for_setting_config_path=["-c", "--cfg", "--config"],
        config_arg_is_required=True,
    )

    parser.add_argument('--model', type=str, required=True, help='Name of the model')
    parser.add_argument('--alpha', type=float, default=.1, help='Name of the model')
    parser.add("--ffn_activation", choices=["relu", "sigmoid"], default="relu",
               help="The activation function to use in the FFN.")
    parser.add("--ffn_hidden_dim", default=256, type=int,
               help="The number of nodes in the FFN hidden layers.")
    parser.add("--ffn_num_layers", default=3, type=int,
               help="The number of layers in the FFN.")
    parser.add("--dropout_prob", default=0, type=float,
               help="The probability by which a node will be dropped out.")
    parser.add("--lr_scheduler", choices=["cosine", "cosine_warmup", "linear", "step"], default="cosine",
               help="The name of the LR scheduler to utilize.")
    parser.add_argument('--batch_size', type=int, default=32, help='Name of the model')
    parser.add_argument('--bias', type=bool, default=True, help='Name of the model')
    parser.add_argument('--max_epochs', type=int, default=1000, help='Name of the model')
    parser.add_argument('--test_size', type=float, default=.2, help='Name of the model')
    parser.add_argument('--model_path', type=str, required=True, help='Name of the model')
    parser.add_argument('--range_size', type=int, required=True, help='Name of the model')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--num_moments', type=int, required=True, help='Number of moments')
    parser.add_argument('--lr', type=float, default=1e-3, help='Number of moments')
    parser.add_argument('--devices', default=-1, help="Input can be an int or a list of ints")
    parser.add_argument('--constraint_weights', type=parse_float_list, required=True, help='List of constraint weights')
    
    args = parser.parse_args()

    return args