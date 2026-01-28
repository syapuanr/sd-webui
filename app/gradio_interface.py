import gradio as gr
import torch
import random
from pathlib import Path
from .model_loader import ModelLoader
from .inference import InferenceEngine
from .memory_manager import MemoryManager

# ============================================================================
# UI Component Functions
# ============================================================================

def create_generation_settings():
    """Create common generation settings UI"""
    with gr.Accordion("Generation Settings", open=True):
        with gr.Row():
            width = gr.Slider(minimum=256, maximum=2048, step=64, value=512, label="Width")
            height = gr.Slider(minimum=256, maximum=2048, step=64, value=512, label="Height")
        with gr.Row():
            steps = gr.Slider(minimum=1, maximum=150, step=1, value=20, label="Sampling Steps")
            cfg_scale = gr.Slider(minimum=1, maximum=30, step=0.5, value=7.5, label="CFG Scale")
        with gr.Row():
            num_images = gr.Slider(minimum=1, maximum=4, step=1, value=1, label="Number of Images")
            seed = gr.Number(value=-1, label="Seed (-1 for random)", precision=0)
        scheduler = gr.Dropdown(
            choices=["DPMSolverMultistep", "Euler", "DDIM"],
            value="DPMSolverMultistep",
            label="Scheduler"
        )
    return width, height, steps, cfg_scale, num_images, seed, scheduler

def create_prompt_inputs():
    """Create prompt input UI"""
    prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here...", lines=3)
    negative_prompt = gr.Textbox(
        label="Negative Prompt",
        placeholder="Enter negative prompt here...",
        lines=2,
        value="blurry, bad quality, worst quality, low resolution"
    )
    return prompt, negative_prompt

def create_model_selector():
    """Create model selection UI"""
    available_models = [
        "runwayml/stable-diffusion-v1-5",
        "stabilityai/stable-diffusion-2-1",
        "stabilityai/stable-diffusion-xl-base-1.0"
    ]
    with gr.Accordion("Model Settings", open=False):
        model_source = gr.Radio(choices=["HuggingFace", "Local File"], value="HuggingFace", label="Model Source")
        model_id = gr.Dropdown(choices=available_models, value=available_models[0], label="HuggingFace Model", allow_custom_value=True, visible=True)
        model_file = gr.File(label="Local Model File", file_types=[".safetensors", ".ckpt"], visible=False)
        vae_file = gr.File(label="Custom VAE (optional)", file_types=[".safetensors", ".pt"])
        load_model_btn = gr.Button("Load Model", variant="primary")
        model_status = gr.Textbox(label="Model Status", value="No model loaded", interactive=False)
    return model_source, model_id, model_file, vae_file, load_model_btn, model_status

def create_lora_settings():
    """Create LoRA settings UI"""
    with gr.Accordion("LoRA Settings", open=False):
        gr.Markdown("### LoRA 1")
        with gr.Row():
            lora1_file = gr.File(label="LoRA File", file_types=[".safetensors", ".pt"])
            lora1_weight = gr.Slider(minimum=-2.0, maximum=2.0, step=0.05, value=1.0, label="Weight")
        gr.Markdown("### LoRA 2")
        with gr.Row():
            lora2_file = gr.File(label="LoRA File", file_types=[".safetensors", ".pt"])
            lora2_weight = gr.Slider(minimum=-2.0, maximum=2.0, step=0.05, value=1.0, label="Weight")
        gr.Markdown("### LoRA 3")
        with gr.Row():
            lora3_file = gr.File(label="LoRA File", file_types=[".safetensors", ".pt"])
            lora3_weight = gr.Slider(minimum=-2.0, maximum=2.0, step=0.05, value=1.0, label="Weight")
    return (lora1_file, lora1_weight, lora2_file, lora2_weight, lora3_file, lora3_weight)

def create_advanced_settings():
    """Create advanced settings UI"""
    with gr.Accordion("Advanced Settings", open=False):
        with gr.Row():
            enable_cpu_offload = gr.Checkbox(label="Enable CPU Offload", value=False)
            enable_xformers = gr.Checkbox(label="Enable xFormers", value=True)
        with gr.Row():
            enable_vae_tiling = gr.Checkbox(label="Enable VAE Tiling", value=True)
            enable_attention_slicing = gr.Checkbox(label="Enable Attention Slicing", value=True)
        clip_skip = gr.Slider(minimum=0, maximum=12, step=1, value=0, label="CLIP Skip")
    return (enable_cpu_offload, enable_xformers, enable_vae_tiling, enable_attention_slicing, clip_skip)

def create_memory_monitor():
    """Create memory monitoring UI"""
    with gr.Accordion("System Monitor", open=False):
        memory_stats = gr.Textbox(label="Memory Statistics", lines=10, interactive=False)
        refresh_memory_btn = gr.Button("Refresh Memory Stats")
    return memory_stats, refresh_memory_btn

# ============================================================================
# Main Gradio Interface Class
# ============================================================================

class GradioInterface:
    """Main Gradio interface for Stable Diffusion WebUI"""
    
    def __init__(self, config):
        self.config = config
        self.model_loader = ModelLoader(config)
        self.inference_engine = InferenceEngine(self.model_loader)
        from .memory_manager import MemoryManager
        self.memory_manager = MemoryManager(config)
        
        # Output directory
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(exist_ok=True)
    
    def load_model_handler(self, model_source, model_id, model_file, vae_file,
                          lora1, lora1_w, lora2, lora2_w, lora3, lora3_w):
        """Handle model loading"""
        try:
            lora_paths = []
            lora_weights = []
            
            for lora_file, weight in [(lora1, lora1_w), (lora2, lora2_w), (lora3, lora3_w)]:
                if lora_file is not None:
                    lora_paths.append(lora_file.name if hasattr(lora_file, 'name') else lora_file)
                    lora_weights.append(weight)
            
            if model_source == "HuggingFace":
                self.model_loader.load_model(
                    model_id=model_id,
                    vae_path=vae_file.name if vae_file else None,
                    lora_paths=lora_paths if lora_paths else None,
                    lora_weights=lora_weights if lora_weights else None
                )
                status = f"✓ Model loaded: {model_id}"
            else:
                if model_file is None:
                    return "✗ Please select a model file"
                
                self.model_loader.load_model(
                    model_path=model_file.name,
                    vae_path=vae_file.name if vae_file else None,
                    lora_paths=lora_paths if lora_paths else None,
                    lora_weights=lora_weights if lora_weights else None
                )
                status = f"✓ Model loaded: {Path(model_file.name).name}"
            
            info = self.model_loader.get_current_model_info()
            status += f"\nType: {info['type'].upper()}"
            status += f"\nDevice: {info['device']}"
            status += f"\nDtype: {info['dtype']}"
            
            return status
            
        except Exception as e:
            return f"✗ Error loading model: {str(e)}"
    
    def generate_txt2img_handler(self, prompt, negative_prompt, width, height,
                                 steps, cfg_scale, num_images, seed, scheduler):
        """Handle text-to-image generation"""
        try:
            if not self.model_loader.current_pipeline:
                return None, "✗ Please load a model first"
            
            if seed == -1:
                seed = random.randint(0, 2147483647)
            
            current_scheduler = self.model_loader.current_pipeline.scheduler.__class__.__name__
            if scheduler not in current_scheduler:
                self.model_loader.set_scheduler(
                    self.model_loader.current_pipeline, 
                    scheduler
                )
            
            images = self.inference_engine.generate_txt2img(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=int(width),
                height=int(height),
                num_inference_steps=int(steps),
                guidance_scale=cfg_scale,
                num_images=int(num_images),
                seed=int(seed)
            )
            
            saved_paths = self.inference_engine.save_images(
                images, 
                output_dir=str(self.output_dir),
                prefix="txt2img"
            )
            
            info = f"✓ Generated {len(images)} image(s)\n"
            info += f"Seed: {seed}\n"
            info += f"Saved to: {self.output_dir}"
            
            return images, info
            
        except Exception as e:
            import traceback
            error_msg = f"✗ Error: {str(e)}\n\n{traceback.format_exc()}"
            return None, error_msg
    
    def generate_img2img_handler(self, init_image, prompt, negative_prompt,
                                strength, steps, cfg_scale, num_images, seed, scheduler):
        """Handle image-to-image generation"""
        try:
            if not self.model_loader.current_pipeline:
                return None, "✗ Please load a model first"
            
            if init_image is None:
                return None, "✗ Please provide an input image"
            
            if seed == -1:
                seed = random.randint(0, 2147483647)
            
            images = self.inference_engine.generate_img2img(
                prompt=prompt,
                image=init_image,
                negative_prompt=negative_prompt,
                strength=strength,
                num_inference_steps=int(steps),
                guidance_scale=cfg_scale,
                num_images=int(num_images),
                seed=int(seed)
            )
            
            saved_paths = self.inference_engine.save_images(
                images,
                output_dir=str(self.output_dir),
                prefix="img2img"
            )
            
            info = f"✓ Generated {len(images)} image(s)\n"
            info += f"Seed: {seed}\n"
            info += f"Strength: {strength}\n"
            info += f"Saved to: {self.output_dir}"
            
            return images, info
            
        except Exception as e:
            import traceback
            error_msg = f"✗ Error: {str(e)}\n\n{traceback.format_exc()}"
            return None, error_msg
    
    def get_memory_stats_handler(self):
        """Get current memory statistics"""
        stats = []
        
        gpu_info = self.memory_manager.get_gpu_memory_info()
        if gpu_info:
            stats.append("=== GPU Memory ===")
            stats.append(f"Allocated: {gpu_info['allocated']:.2f} GB")
            stats.append(f"Reserved:  {gpu_info['reserved']:.2f} GB")
            stats.append(f"Free:      {gpu_info['free']:.2f} GB")
            stats.append(f"Total:     {gpu_info['total']:.2f} GB")
            stats.append("")
        
        ram_info = self.memory_manager.get_ram_info()
        stats.append("=== System RAM ===")
        stats.append(f"Used:      {ram_info['used']:.2f} GB")
        stats.append(f"Available: {ram_info['available']:.2f} GB")
        stats.append(f"Total:     {ram_info['total']:.2f} GB")
        stats.append(f"Usage:     {ram_info['percent']:.1f}%")
        stats.append("")
        
        if self.model_loader.current_pipeline:
            info = self.model_loader.get_current_model_info()
            stats.append("=== Current Model ===")
            stats.append(f"Model: {info['model']}")
            stats.append(f"Type:  {info['type'].upper()}")
            stats.append(f"Device: {info['device']}")
        else:
            stats.append("=== Current Model ===")
            stats.append("No model loaded")
        
        return "\n".join(stats)
    
    def update_model_input_visibility(self, source):
        """Update visibility of model input fields"""
        if source == "HuggingFace":
            return gr.update(visible=True), gr.update(visible=False)
        else:
            return gr.update(visible=False), gr.update(visible=True)
    
    def create_interface(self):
        """Create the complete Gradio interface"""
        
        with gr.Blocks(title="Stable Diffusion WebUI", theme=gr.themes.Soft()) as demo:
            
            gr.Markdown("# 🎨 Stable Diffusion WebUI\n### Optimized for Google Colab")
            
            # Model Selection
            (model_source, model_id, model_file, vae_file, 
             load_model_btn, model_status) = create_model_selector()
            
            # LoRA Settings
            (lora1_file, lora1_weight, lora2_file, lora2_weight,
             lora3_file, lora3_weight) = create_lora_settings()
            
            # Advanced Settings
            (enable_cpu_offload, enable_xformers, enable_vae_tiling,
             enable_attention_slicing, clip_skip) = create_advanced_settings()
            
            # Memory Monitor
            memory_stats, refresh_memory_btn = create_memory_monitor()
            
            # Main Tabs
            with gr.Tabs():
                # Text-to-Image
                with gr.Tab("Text-to-Image"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            prompt_t2i, negative_prompt_t2i = create_prompt_inputs()
                            (width_t2i, height_t2i, steps_t2i, cfg_t2i,
                             num_images_t2i, seed_t2i, scheduler_t2i) = create_generation_settings()
                            generate_t2i_btn = gr.Button("Generate", variant="primary", size="lg")
                        
                        with gr.Column(scale=1):
                            output_gallery_t2i = gr.Gallery(label="Generated Images", columns=2, rows=2)
                            output_info_t2i = gr.Textbox(label="Generation Info", lines=5, interactive=False)
                
                # Image-to-Image
                with gr.Tab("Image-to-Image"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            init_image = gr.Image(label="Input Image", type="pil")
                            prompt_i2i, negative_prompt_i2i = create_prompt_inputs()
                            strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.75, label="Denoising Strength")
                            (width_i2i, height_i2i, steps_i2i, cfg_i2i,
                             num_images_i2i, seed_i2i, scheduler_i2i) = create_generation_settings()
                            generate_i2i_btn = gr.Button("Generate", variant="primary", size="lg")
                        
                        with gr.Column(scale=1):
                            output_gallery_i2i = gr.Gallery(label="Generated Images", columns=2, rows=2)
                            output_info_i2i = gr.Textbox(label="Generation Info", lines=5, interactive=False)
            
            # Event Handlers
            model_source.change(fn=self.update_model_input_visibility, inputs=[model_source], outputs=[model_id, model_file])
            
            load_model_btn.click(
                fn=self.load_model_handler,
                inputs=[model_source, model_id, model_file, vae_file, lora1_file, lora1_weight, lora2_file, lora2_weight, lora3_file, lora3_weight],
                outputs=[model_status]
            )
            
            generate_t2i_btn.click(
                fn=self.generate_txt2img_handler,
                inputs=[prompt_t2i, negative_prompt_t2i, width_t2i, height_t2i, steps_t2i, cfg_t2i, num_images_t2i, seed_t2i, scheduler_t2i],
                outputs=[output_gallery_t2i, output_info_t2i]
            )
            
            generate_i2i_btn.click(
                fn=self.generate_img2img_handler,
                inputs=[init_image, prompt_i2i, negative_prompt_i2i, strength, steps_i2i, cfg_i2i, num_images_i2i, seed_i2i, scheduler_i2i],
                outputs=[output_gallery_i2i, output_info_i2i]
            )
            
            refresh_memory_btn.click(fn=self.get_memory_stats_handler, outputs=[memory_stats])
            demo.load(fn=self.get_memory_stats_handler, outputs=[memory_stats])
        
        return demo

def create_interface(config):
    """Factory function to create Gradio interface"""
    interface = GradioInterface(config)
    return interface.create_interface()