import torch
import os
from pathlib import Path
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    DDIMScheduler,
    AutoencoderKL
)
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from .memory_manager import MemoryManager

class ModelLoader:
    """Loads and manages Stable Diffusion models"""
    
    # Supported schedulers
    SCHEDULERS = {
        'DPMSolverMultistep': DPMSolverMultistepScheduler,
        'Euler': EulerAncestralDiscreteScheduler,
        'DDIM': DDIMScheduler,
    }
    
    def __init__(self, config):
        self.config = config
        self.device = config.get('system.device', 'cuda')
        self.dtype = getattr(torch, config.get('system.dtype', 'float16'))
        self.memory_manager = MemoryManager(config)
        
        self.current_model = None
        self.current_pipeline = None
        self.model_type = None
        
    def detect_model_type(self, model_path):
        """Detect if model is SD 1.5, SD 2.x, or SDXL"""
        # This is a simplified detection
        # In production, you'd check config files
        if 'xl' in model_path.lower() or 'sdxl' in model_path.lower():
            return 'sdxl'
        elif 'sd2' in model_path.lower() or '2-1' in model_path.lower():
            return 'sd2'
        else:
            return 'sd15'
    
    def get_pipeline_class(self, model_type):
        """Get appropriate pipeline class for model type"""
        if model_type == 'sdxl':
            return StableDiffusionXLPipeline
        else:
            return StableDiffusionPipeline
    
    def load_model_from_single_file(self, checkpoint_path, model_type=None):
        """Load model from .safetensors or .ckpt file"""
        print(f"\nLoading model from: {checkpoint_path}")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model file not found: {checkpoint_path}")
        
        # Auto-detect model type if not specified
        if model_type is None:
            model_type = self.detect_model_type(checkpoint_path)
        
        print(f"Detected model type: {model_type.upper()}")
        
        # Get appropriate pipeline class
        PipelineClass = self.get_pipeline_class(model_type)
        
        # Load pipeline from single file
        try:
            if checkpoint_path.endswith('.safetensors'):
                pipe = PipelineClass.from_single_file(
                    checkpoint_path,
                    torch_dtype=self.dtype,
                    use_safetensors=True,
                    load_safety_checker=False
                )
            else:
                pipe = PipelineClass.from_single_file(
                    checkpoint_path,
                    torch_dtype=self.dtype,
                    load_safety_checker=False
                )
            
            print("✓ Model loaded successfully")
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise
        
        return pipe, model_type
    
    def load_model_from_hub(self, model_id, model_type=None):
        """Load model from Hugging Face Hub"""
        print(f"\nLoading model from HuggingFace: {model_id}")
        
        # Auto-detect model type if not specified
        if model_type is None:
            model_type = self.detect_model_type(model_id)
        
        PipelineClass = self.get_pipeline_class(model_type)
        
        try:
            pipe = PipelineClass.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                use_safetensors=True,
                safety_checker=None
            )
            print("✓ Model loaded from HuggingFace")
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise
        
        return pipe, model_type
    
    def load_vae(self, vae_path):
        """Load custom VAE"""
        if not vae_path or not os.path.exists(vae_path):
            return None
        
        print(f"Loading VAE from: {vae_path}")
        
        try:
            vae = AutoencoderKL.from_single_file(
                vae_path,
                torch_dtype=self.dtype
            )
            print("✓ VAE loaded successfully")
            return vae
        except Exception as e:
            print(f"⚠ Error loading VAE: {e}")
            return None
    
    def load_lora(self, pipe, lora_path, lora_weight=1.0):
        """Load LoRA weights"""
        if not os.path.exists(lora_path):
            print(f"⚠ LoRA file not found: {lora_path}")
            return pipe
        
        print(f"Loading LoRA: {lora_path} (weight: {lora_weight})")
        
        try:
            pipe.load_lora_weights(lora_path)
            pipe.fuse_lora(lora_scale=lora_weight)
            print("✓ LoRA loaded successfully")
        except Exception as e:
            print(f"⚠ Error loading LoRA: {e}")
        
        return pipe
    
    def set_scheduler(self, pipe, scheduler_name):
        """Set scheduler for the pipeline"""
        if scheduler_name not in self.SCHEDULERS:
            print(f"⚠ Unknown scheduler: {scheduler_name}, using default")
            return pipe
        
        SchedulerClass = self.SCHEDULERS[scheduler_name]
        pipe.scheduler = SchedulerClass.from_config(pipe.scheduler.config)
        print(f"✓ Scheduler set to: {scheduler_name}")
        
        return pipe
    
    def load_model(self, model_path=None, model_id=None, vae_path=None, 
                   lora_paths=None, lora_weights=None, scheduler=None):
        """
        Main method to load complete model with all components
        
        Args:
            model_path: Path to local .safetensors/.ckpt file
            model_id: HuggingFace model ID
            vae_path: Path to custom VAE
            lora_paths: List of LoRA file paths
            lora_weights: List of LoRA weights (default 1.0)
            scheduler: Scheduler name
        """
        self.memory_manager.clear_cache()
        
        # Load base model
        if model_path:
            pipe, model_type = self.load_model_from_single_file(model_path)
        elif model_id:
            pipe, model_type = self.load_model_from_hub(model_id)
        else:
            # Load default model
            default_model = "runwayml/stable-diffusion-v1-5"
            print(f"No model specified, loading default: {default_model}")
            pipe, model_type = self.load_model_from_hub(default_model)
        
        # Load custom VAE if specified
        if vae_path:
            vae = self.load_vae(vae_path)
            if vae:
                pipe.vae = vae
        
        # Load LoRAs if specified
        if lora_paths:
            if not isinstance(lora_paths, list):
                lora_paths = [lora_paths]
            
            if lora_weights is None:
                lora_weights = [1.0] * len(lora_paths)
            elif not isinstance(lora_weights, list):
                lora_weights = [lora_weights]
            
            for lora_path, lora_weight in zip(lora_paths, lora_weights):
                pipe = self.load_lora(pipe, lora_path, lora_weight)
        
        # Set scheduler
        if scheduler:
            pipe = self.set_scheduler(pipe, scheduler)
        else:
            default_scheduler = self.config.get('generation.scheduler', 'DPMSolverMultistep')
            pipe = self.set_scheduler(pipe, default_scheduler)
        
        # Move to device
        pipe = pipe.to(self.device)
        
        # Apply memory optimizations
        pipe = self.memory_manager.optimize_pipeline(pipe)
        
        # Store current state
        self.current_pipeline = pipe
        self.current_model = model_path or model_id
        self.model_type = model_type
        
        # Print memory stats
        self.memory_manager.print_memory_stats()
        
        print(f"✓ Model ready: {self.current_model}")
        print(f"✓ Model type: {model_type.upper()}\n")
        
        return pipe
    
    def unload_model(self):
        """Unload current model to free memory"""
        if self.current_pipeline:
            del self.current_pipeline
            self.current_pipeline = None
            self.current_model = None
            self.model_type = None
            
            self.memory_manager.clear_cache()
            print("✓ Model unloaded")
    
    def get_current_model_info(self):
        """Get information about currently loaded model"""
        return {
            'model': self.current_model,
            'type': self.model_type,
            'device': self.device,
            'dtype': str(self.dtype),
            'loaded': self.current_pipeline is not None
        }