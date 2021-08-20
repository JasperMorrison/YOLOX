#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os,sys
from loguru import logger

sys.path.append('./PytorchToCaffe')
from PytorchToCaffe import pytorch_to_caffe

import torch
from torch import nn

from yolox.exp import get_exp
from yolox.models.network_blocks import SiLU
from yolox.utils import replace_module, fuse_model

def make_parser():
    parser = argparse.ArgumentParser("YOLOX onnx deploy")
    parser.add_argument(
        "--output-name", type=str, default="yolox.onnx", help="output name of models"
    )
    parser.add_argument("--input", default="images", type=str, help="input name of onnx model")
    parser.add_argument("--output", default="output", type=str, help="output name of onnx model")

    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="expriment description file",
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt path")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    return parser


@logger.catch
def main():
    args = make_parser().parse_args()
    logger.info("args value: {}".format(args))
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    model = exp.get_model()
    if args.ckpt is None:
        file_name = os.path.join(exp.output_dir, args.experiment_name)
        ckpt_file = os.path.join(file_name, "best_ckpt.pth.tar")
    else:
        ckpt_file = args.ckpt

    ckpt = torch.load(ckpt_file, map_location="cpu")
    # load the model state dict

    model.eval()
    if "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt, strict=False)
    model = replace_module(model, nn.SiLU, SiLU)
    model.head.decode_in_inference = False
    model = fuse_model(model)
    model.head.forward = model.head.forward_caffe

    logger.info("loaded checkpoint done.")
    dummy_input = torch.randn(1, 3, exp.test_size[0], exp.test_size[1])

    # caffe export
    try:
        print('\nStarting caffe export with torch %s...' % torch.__version__)
        os.makedirs('./temp', exist_ok=True)
        name = "yolox"
        pytorch_to_caffe.trans_net(model,dummy_input, name)
        pytorch_to_caffe.save_prototxt('temp/{}.prototxt'.format(name))
        pytorch_to_caffe.save_caffemodel('temp/{}.caffemodel'.format(name))
        print('caffe export success, saved as %s' % 'temp/{}.caffemodel'.format(name))
    except Exception as e:
        print('caffe export failure: %s' % e)

if __name__ == "__main__":
    main()
