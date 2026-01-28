import gc
import torch
import psutil
from typing import Optional

class MemoryManager:
    """Manages GPU and RAM memory to prevent crashes"""
    
    def __init__(self, config):
        self.config = config
        self.device = config.get('system.device', 'cuda')
        
    def clear_cache(self):
        """Clear GPU cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
        
    def get_gpu_memory_info(self):
        """Get current GPU memory usage"""
        if not torch.cuda.is_available():
            return None
            
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        return {
            'allocated': allocated,
            'reserved': reserved,
            'total': total,
            'free': total - reserved
        }
    
    def get_ram_info(self):
        """Get system RAM usage"""
        ram = psutil.virtual_memory()
        return {
            'total': ram.total / 1024**3,  # GB
            'available': ram.available / 1024**3,
            'percent': ram.percent,
            'used': ram.used / 1024**3
        }
    
    def print_memory_stats(self):
        """Print current memory statistics"""
        print("\n" + "="*60)
        print("MEMORY STATISTICS")
        print("="*60)
        
        # GPU Memory
        gpu_info = self.get_gpu_memory_info()
        if gpu_info:
            print(f"GPU Memory:")
            print(f"  Allocated: {gpu_info['allocated']:.2f} GB")
            print(f"  Reserved:  {gpu_info['reserved']:.2f} GB")
            print(f"  Free:      {gpu_info['free']:.2f} GB")
            print(f"  Total:     {gpu_info['total']:.2f} GB")
        
        # RAM
        ram_info = self.get_ram_info()
        print(f"\nRAM:")
        print(f"  Used:      {ram_info['used']:.2f} GB")
        print(f"  Available: {ram_info['available']:.2f} GB")
        print(f"  Total:     {ram_info['total']:.2f} GB")
        print(f"  Usage:     {ram_info['percent']:.1f}%")
        print("="*60 + "\n")
    
    def enable_memory_efficient_attention(self, pipe):
        """Enable memory efficient attention mechanisms"""
        try:
            # Enable xFormers if available
            if self.config.get('system.xformers_enabled', True):
                pipe.enable_xformers_memory_efficient_attention()
                print("✓ xFormers memory efficient attention enabled")
        except Exception as e:
            print(f"⚠ Could not enable xFormers: {e}")
            
        # Enable attention slicing
        if self.config.get('system.attention_slicing', True):
            pipe.enable_attention_slicing(1)
            print("✓ Attention slicing enabled")
            
        # Enable VAE slicing for large images
        if self.config.get('memory.vae_tiling', True):
            try:
                pipe.enable_vae_slicing()
                print("✓ VAE slicing enabled")
            except:
                pass
                
        # Enable VAE tiling
        try:
            pipe.enable_vae_tiling()
            print("✓ VAE tiling enabled")
        except:
            pass
            
        return pipe
    
    def enable_model_cpu_offload(self, pipe):
        """Enable model CPU offload to save GPU memory"""
        if self.config.get('memory.enable_model_cpu_offload', False):
            try:
                pipe.enable_model_cpu_offload()
                print("✓ Model CPU offload enabled")
            except Exception as e:
                print(f"⚠ Could not enable CPU offload: {e}")
        
        return pipe
    
    def enable_sequential_cpu_offload(self, pipe):
        """Enable sequential CPU offload (most aggressive memory saving)"""
        if self.config.get('memory.enable_sequential_cpu_offload', False):
            try:
                pipe.enable_sequential_cpu_offload()
                print("✓ Sequential CPU offload enabled")
            except Exception as e:
                print(f"⚠ Could not enable sequential CPU offload: {e}")
        
        return pipe
    
    def optimize_pipeline(self, pipe):
        """Apply all memory optimizations to pipeline"""
        print("\nApplying memory optimizations...")
        
        pipe = self.enable_memory_efficient_attention(pipe)
        pipe = self.enable_model_cpu_offload(pipe)
        
        # Don't use both CPU offload methods together
        if not self.config.get('memory.enable_model_cpu_offload', False):
            pipe = self.enable_sequential_cpu_offload(pipe)
        
        self.clear_cache()
        print("✓ Memory optimization complete\n")
        
        return pipe