import torch

from tqdm.auto import trange

from diffusion.utils import t_to_alpha_sigma
from k_diffusion.external import VDenoiser

from typing import Tuple, Callable
from diffusion_library.scheduler import SchedulerType
from diffusion_library.sampler import SamplerType
from dance_diffusion.base.model import ModelWrapperBase
from dance_diffusion.base.inference import InferenceBase

from util.util import tensor_slerp_2D, PosteriorSampling
    
class LDDInference(InferenceBase):
    
    def __init__(
        self,
        device_accelerator: torch.device = None,
        device_offload: torch.device = None,
        optimize_memory_use: bool = False,
        use_autocast: bool = True,
        model: ModelWrapperBase = None
    ):
        super().__init__(device_accelerator, device_offload, optimize_memory_use, use_autocast, model)
        
    def generate(
        self,
        callback: Callable = None,
        batch_size: int = None,
        seed: int = None,
        steps: int = None,
        scheduler: SchedulerType = None,
        scheduler_args: dict = None,
        sampler: SamplerType = None,
        sampler_args: dict = None,
        **kwargs
    ):
        self.generator.manual_seed(seed)
        
        step_list = scheduler.get_step_list(steps, self.device_accelerator.type, **scheduler_args)#step_list = step_list[:-1] if sampler in [SamplerType.V_PRK, SamplerType.V_PLMS, SamplerType.V_PIE, SamplerType.V_PLMS2, SamplerType.V_IPLMS] else step_list
        
        if SamplerType.is_v_sampler(sampler):
            x_T = torch.randn([batch_size, self.model.latent_dim, self.model.native_chunk_size // self.model.downsampling_ratio], generator=self.generator, device=self.device_accelerator)
            model = self.model.diffusion
        else:
            x_T = step_list[0] * torch.randn([batch_size, self.model.latent_dim, self.model.native_chunk_size // self.model.downsampling_ratio], generator=self.generator, device=self.device_accelerator)
            model = VDenoiser(self.model.diffusion)
        
        with self.offload_context(self.model.diffusion):
            x_0 = sampler.sample(
                model,
                x_T,
                step_list,
                callback,
                **sampler_args
            )
            
        with self.offload_context(self.model.ae_decoder):
            return self.model.ae_decoder(x_0).float()