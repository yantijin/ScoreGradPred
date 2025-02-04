import argparse
import requests
import json
import time
import logging
import hmac
import urllib
import base64
import hashlib

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='elec', help='dataset names')
    parser.add_argument('--name', type=str, default='ddpm_d', help='method for sde')
    parser.add_argument('--beta1', type=float, default=0.01, help='beta min')
    parser.add_argument('--beta2', type=float, default=10, help='beta max')
    parser.add_argument('--scale', type=int, default=100, help='num scales')
    parser.add_argument('--epochs', type=int, default=10, help='num of epochs')
    parser.add_argument('--batch', type=int, default=32, help='num of epochs')
    args = parser.parse_args()
    args.path = './metrics_' + args.data + '.log'
    return args

def parse_args_lisde():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='elec', help='dataset names')
    parser.add_argument('--name', type=str, default='ddpm_d', help='method for sde')
    parser.add_argument('--sigma', type=float, default=1e-6, help='sigma')
    parser.add_argument('--eps', type=float, default=1e-3, help='eps')
    parser.add_argument('--clamp_val', type=float, default=2., help='truncation range')
    parser.add_argument('--scale', type=int, default=20, help='num scales')
    parser.add_argument('--epochs', type=int, default=10, help='num of epochs')
    parser.add_argument('--batch', type=int, default=32, help='num of epochs')
    args = parser.parse_args()
    args.path = './metrics_' + args.data + '.log'
    return args


def write_to_file(args, config, metrics, path):
    with open(path, 'a+', encoding='utf8') as file_obj:
        file_obj.write('='*20)
        file_obj.write(str(args))
        file_obj.write('=' * 20)
        file_obj.write('\n')
        file_obj.write(str(config))
        file_obj.write('\n')
        file_obj.write('\n'.join([str((term, metrics[term])) for term in metrics]))
        file_obj.write('\n')
