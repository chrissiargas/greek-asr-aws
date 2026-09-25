import argparse
import os.path
import yaml
from os.path import dirname, abspath
from typing import Optional

class Args:
    def __init__(self):
        pass

    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(f"Key {key} not found in Arguments")

class Parser:

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="pre-processing and training parameters"
        )

    def __call__(self, exp: int = 0, *args, **kwargs):
        project_root = dirname(abspath(__file__))

        if exp == 0:
            config_path = os.path.join(project_root, 'my_config.yaml')
        else:
            config_path = os.path.join(project_root, f'my_config_exp{exp}.yaml')

        self.parser.add_argument(
            '--config',
            default=config_path,
            help='config file location'
        )

        self.parser.add_argument(
            '--data_args',
            default=dict(),
            type=dict,
            help='data arguments'
        )

        self.parser.add_argument(
            '--main_args',
            default=dict(),
            type=dict,
            help='pre_processing & training arguments'
        )

        self.parser.add_argument(
            '--dual_fuse_args',
            default=dict(),
            type=dict,
            help='Dual Fusion Arguments'
        )

        self.parser.add_argument(
            '--eval_args',
            default=dict(),
            type=dict,
            help='Evaluation Arguments'
        )

    def as_str(self, theme: Optional[str] = None, learning: str = 'sl', with_event: bool = True) -> str:

        if theme is None:
            out = f'{self.datasets}'

        return out


    def get_args(self, exp: int = 0):
        self.__call__(exp=exp)
        args = self.parser.parse_args(args=[])
        configFile = args.config

        assert configFile is not None

        with open(configFile, 'r') as cf:
            defaultArgs = yaml.load(cf, Loader=yaml.FullLoader)

        keys = vars(args).keys()

        for defaultKey in defaultArgs.keys():
            if defaultKey not in keys:
                print('WRONG ARG: {}'.format(defaultKey))
                assert (defaultKey in keys)

        self.parser.set_defaults(**defaultArgs)
        args = self.parser.parse_args(args=[])

        self.language = args.data_args['language']
        self.tmpdir = args.data_args['tmpdir']
        self.hf_cache = args.data_args['hf_cache']
        self.dataset_path = args.data_args['dataset_path']
        self.sampling_rate = args.data_args['sampling_rate']
        self.other_to_train = args.data_args['other_to_train']

        self.datasets = args.main_args['datasets']
        self.split_type = args.main_args['split_type']
        self.splitting = args.main_args['splitting']
        self.results_path = args.main_args['results_path']
        self.checkpoint_path = args.main_args['checkpoint_path']

        self.dual_fuse_args = Args()
        self.dual_fuse_args.checkpoint_path = args.dual_fuse_args['checkpoint_path']

        ## Data Configurations
        self.dual_fuse_args.randomize = args.dual_fuse_args['randomize']
        self.dual_fuse_args.normalize = args.dual_fuse_args['normalize']
        self.dual_fuse_args.micro_data = args.dual_fuse_args['micro_data']
        self.dual_fuse_args.micro_size = args.dual_fuse_args['micro_size']
        self.dual_fuse_args.norm_mono = args.dual_fuse_args['norm_mono']
        self.dual_fuse_args.interleave_temperature = args.dual_fuse_args['interleave_temperature']

        ## Regularization Configurations
        self.dual_fuse_args.blank_training = args.dual_fuse_args['blank_training']
        self.dual_fuse_args.audio_dropout = args.dual_fuse_args['audio_dropout']
        self.dual_fuse_args.text_perturbation = args.dual_fuse_args['text_perturbation']
        self.dual_fuse_args.text_dropout = args.dual_fuse_args['text_dropout']
        self.dual_fuse_args.spec_augment = args.dual_fuse_args['spec_augment']

        ## Input Injection Configurations
        self.dual_fuse_args.include_adapter = args.dual_fuse_args['include_adapter']
        self.dual_fuse_args.downsample_K = args.dual_fuse_args['downsample_K']
        self.dual_fuse_args.input_downsample = args.dual_fuse_args['input_downsample']
        self.dual_fuse_args.hidden_dim = args.dual_fuse_args['hidden_dim']
        self.dual_fuse_args.static_projector = args.dual_fuse_args['static_projector']

        ## Cross-Attention Injection Configurations
        self.dual_fuse_args.static_injection_layers = args.dual_fuse_args['static_injection_layers']
        self.dual_fuse_args.injection_layers = args.dual_fuse_args['injection_layers']
        self.dual_fuse_args.pyramid_layers = args.dual_fuse_args['pyramid_layers']
        self.dual_fuse_args.downsample_L = args.dual_fuse_args['downsample_L']
        self.dual_fuse_args.injection_downsample = args.dual_fuse_args['injection_downsample']
        self.dual_fuse_args.causal_fusion = args.dual_fuse_args['causal_fusion']
        self.dual_fuse_args.gated_cross_attention = args.dual_fuse_args['gated_cross_attention']
        self.dual_fuse_args.downsamplers = args.dual_fuse_args['downsamplers']
        self.dual_fuse_args.positional_info = args.dual_fuse_args['positional_info']
        self.dual_fuse_args.layer_wise_fusion = args.dual_fuse_args['layer_wise_fusion']
        self.dual_fuse_args.layer_weights_static = args.dual_fuse_args['layer_weights_static']

        ## LoRA Configurations
        self.dual_fuse_args.linguistic_lora = args.dual_fuse_args['linguistic_lora']
        self.dual_fuse_args.acoustic_lora = args.dual_fuse_args['acoustic_lora']
        self.dual_fuse_args.lora_params = args.dual_fuse_args['lora_params']
        self.dual_fuse_args.lora_r = args.dual_fuse_args['lora_r']
        self.dual_fuse_args.two_stage = args.dual_fuse_args['two_stage']
        self.dual_fuse_args.lora_lr = args.dual_fuse_args['lora_lr']
        self.dual_fuse_args.proj_lr = args.dual_fuse_args['proj_lr']

        ## Auxiliary Losses Configurations
        self.dual_fuse_args.predict_duration = args.dual_fuse_args['predict_duration']
        self.dual_fuse_args.duration_resolution = args.dual_fuse_args['duration_resolution']
        self.dual_fuse_args.max_duration = args.dual_fuse_args['max_duration']
        self.dual_fuse_args.ctc = args.dual_fuse_args['ctc']
        self.dual_fuse_args.audio_forecasting = args.dual_fuse_args['audio_forecasting']
        self.dual_fuse_args.ctc_weight = args.dual_fuse_args['ctc_weight']
        self.dual_fuse_args.audio_weight = args.dual_fuse_args['audio_weight']
        self.dual_fuse_args.duration_weight = args.dual_fuse_args['duration_weight']

        ## Training Configurations
        self.dual_fuse_args.first_stage_epochs = args.dual_fuse_args['first_stage_epochs']
        self.dual_fuse_args.second_stage_epochs = args.dual_fuse_args['second_stage_epochs']
        self.dual_fuse_args.early_stopping_patience = args.dual_fuse_args['early_stopping_patience']

        ## Prompt Configurations
        self.dual_fuse_args.prompt_persona = args.dual_fuse_args['prompt_persona']
        self.dual_fuse_args.prompt_instruction = args.dual_fuse_args['prompt_instruction']
        self.dual_fuse_args.prompt_verbatim = args.dual_fuse_args['prompt_verbatim']

        self.dual_fuse_args.training_args = args.dual_fuse_args['training_args']

        self.eval_args = Args()
        self.eval_args.training_args = args.eval_args['training_args']

if __name__ == "__main__":
    parser = Parser()
    parser.get_args()