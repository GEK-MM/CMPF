"""RISBench 按表达式长度分桶评测。"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from length_eval_common import run


def parse_args():
    parser = argparse.ArgumentParser(description='Length-bucket eval on RISBench.')
    parser.add_argument('--weights', default='saved_models/couple_decoder/coupe_de_risbench_best.pth')
    parser.add_argument('--data-root', default='./refer')
    parser.add_argument('--bert-tokenizer', default='/home/ubuntu/glk/code/bert-base-uncased/')
    parser.add_argument('--img-size', default=512, type=int)
    parser.add_argument('--num-query', default=20, type=int)
    parser.add_argument('--splits', nargs='+', default=['val', 'test'])
    parser.add_argument('--output', default='logs/length_eval_risbench.json')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    os.chdir('/home/ubuntu/glk/code/private-code/any_try/InferSeg/Lab2')
    run(
        dataset='risbench',
        weights=args.weights,
        splits=args.splits,
        data_root=args.data_root,
        bert_tokenizer=args.bert_tokenizer,
        img_size=args.img_size,
        num_query=args.num_query,
        output=args.output,
    )
