import argparse

parser = argparse.ArgumentParser(description='OUR', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# setting
parser.add_argument('--name', type=str, default="cite")
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--layers', type=int, default=1)
parser.add_argument('--dims', type=int, default=[500], help='Number of units in hidden layer 1.')
parser.add_argument('--epoch', type=int, default=400)
parser.add_argument('--show_training_details', type=bool, default=False)

# clustering performance: acc, nmi, ari, f1
parser.add_argument('--acc', type=float, default=0)
parser.add_argument('--nmi', type=float, default=0)
parser.add_argument('--ari', type=float, default=0)
parser.add_argument('--f1', type=float, default=0)

args = parser.parse_args()

