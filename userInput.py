import argparse

parser =argparse.ArgumentParser()
# parser.add_argument("-w", "-weight")
group1 = parser.add_mutually_exclusive_group()
group1.add_argument("-tr", "--train", help="Training", action='store_true')
group1.add_argument("-te", "--test", help="Testing", action='store_true')
args = parser.parse_args()
