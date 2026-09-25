import sys
import os
from os.path import dirname
sys.path.insert(0, dirname(dirname(os.path.abspath(__file__))))
from parser import Parser

def set_environment():
    conf = Parser()
    conf.get_args()

    os.environ['HF_HOME'] = conf.hf_cache
    os.environ['HF_DATASETS_CACHE'] = os.path.join(os.environ['HF_HOME'], 'datasets')
    os.environ['TMPDIR'] = conf.tmpdir