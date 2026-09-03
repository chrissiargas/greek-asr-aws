import sys
import os
from os.path import dirname
sys.path.insert(0, dirname(dirname(os.path.abspath(__file__))))

from config.parser import Parser


def set_environment():
    conf = Parser()
    conf.get_args()

    os.environ['HF_HOME'] = conf.hf_cache
    os.environ['HF_DATASETS_CACHE'] = os.path.join(os.environ['HF_HOME'], 'datasets')
    os.environ['TMPDIR'] = conf.tmpdir

    os.environ['WANDB_API_KEY']="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"

if __name__ == '__main__':
    set_environment()

    from datasets import config

    print(f'Cache Dir: {config.HF_DATASETS_CACHE}')
    from huggingface_hub import constants

    print(f'Hub Dir: {constants.HF_HUB_CACHE}')