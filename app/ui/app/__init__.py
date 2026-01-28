import gradio as gr

def create_generation_settings():
    """Create common generation settings UI"""
    with gr.Accordion("Generation Settings", open=True):
        with gr.Row():
            width = gr.Slider(
                minimum=256,
                maximum=2048,
                step=64,
                value=512,
                label="Width"
            )
            height = gr.Slider(
                minimum=256,
                maximum=2048,
                step=64,
                value=512,
                label="Height"
            )
        
        with gr.Row():
            steps = gr.Slider(
                minimum=1,
                maximum=150,
                step=1,
                value=20,
                label="Sampling Steps"
            )
            cfg_scale = gr.Slider(
                minimum=1,
                maximum=30,
                step=0.5,
                value=7.5,
                label="CFG Scale"
            )
        
        with gr.Row():
            num_images = gr.Slider(
                minimum=1,
                maximum=4,
                step=1,
                value=1,
                label="Number of Images"
            )
            seed = gr.Number(
                value=-1,
                label="Seed (-1 for random)",
                precision=0
            )
        
        scheduler = gr.Dropdown(
            choices=["DPMSolverMultistep", "Euler", "DDIM"],
            value="DPMSolverMultistep",
            label="Scheduler"
        )
    
    return width, height, steps, cfg_scale, num_images, seed, scheduler

def create_prompt_inputs():
    """Create prompt input UI"""
    prompt = gr.Textbox(
        label="Prompt",
        placeholder="Enter your prompt here...",
        lines=3
    )
    
    negative_prompt = gr.Textbox(
        label="Negative Prompt",
        placeholder="Enter negative prompt here...",
        lines=2,
        value="blurry, bad quality, worst quality, low resolution"
    )
    
    return prompt, negative_prompt

def create_model_selector(available_models=None):
    """Create model selection UI"""
    if available_models is None:
        available_models = [
            "runwayml/stable-diffusion-v1-5",
            "stabilityai/stable-diffusion-2-1",
            "stabilityai/stable-diffusion-xl-base-1.0"
        ]
    
    with gr.Accordion("Model Settings", open=False):
        model_source = gr.Radio(
            choices=["HuggingFace", "Local File"],
            value="HuggingFace",
            label="Model Source"
        )
        
        model_id = gr.Dropdown(
            choices=available_models,
            value=available_models[0],
            label="HuggingFace Model",
            allow_custom_value=True,
            visible=True
        )
        
        model_file = gr.File(
            label="Local Model File (.safetensors or .ckpt)",
            file_types=[".safetensors", ".ckpt"],
            visible=False
        )
        
        vae_file = gr.File(
            label="Custom VAE (optional)",
            file_types=[".safetensors", ".pt"],
            visible=True
        )
        
        load_model_btn = gr.Button("Load Model", variant="primary")
        model_status = gr.Textbox(
            label="Model Status",
            value="No model loaded",
            interactive=False
        )
    
    return model_source, model_id, model_file, vae_file, load_model_btn, model_status

def create_lora_settings():
    """Create LoRA settings UI"""
    with gr.Accordion("LoRA Settings", open=False):
        gr.Markdown("### LoRA 1")
        with gr.Row():
            lora1_file = gr.File(
                label="LoRA File",
                file_types=[".safetensors", ".pt"]
            )
            lora1_weight = gr.Slider(
                minimum=-2.0,
                maximum=2.0,
                step=0.05,
                value=1.0,
                label="Weight"
            )
        
        gr.Markdown("### LoRA 2")
        with gr.Row():
            lora2_file = gr.File(
                label="LoRA File",
                file_types=[".safetensors", ".pt"]
            )
            lora2_weight = gr.Slider(
                minimum=-2.0,
                maximum=2.0,
                step=0.05,
                value=1.0,
                label="Weight"
            )
        
        gr.Markdown("### LoRA 3")
        with gr.Row():
            lora3_file = gr.File(
                label="LoRA File",
                file_types=[".safetensors", ".pt"]
            )
            lora3_weight = gr.Slider(
                minimum=-2.0,
                maximum=2.0,
                step=0.05,
                value=1.0,
                label="Weight"
            )
    
    return (lora1_file, lora1_weight, lora2_file, lora2_weight, 
            lora3_file, lora3_weight)

def create_advanced_settings():
    """Create advanced settings UI"""
    with gr.Accordion("Advanced Settings", open=False):
        with gr.Row():
            enable_cpu_offload = gr.Checkbox(
                label="Enable CPU Offload (saves VRAM)",
                value=False
            )
            enable_xformers = gr.Checkbox(
                label="Enable xFormers",
                value=True
            )
        
        with gr.Row():
            enable_vae_tiling = gr.Checkbox(
                label="Enable VAE Tiling",
                value=True
            )
            enable_attention_slicing = gr.Checkbox(
                label="Enable Attention Slicing",
                value=True
            )
        
        clip_skip = gr.Slider(
            minimum=0,
            maximum=12,
            step=1,
            value=0,
            label="CLIP Skip"
        )
    
    return (enable_cpu_offload, enable_xformers, enable_vae_tiling, 
            enable_attention_slicing, clip_skip)

def create_memory_monitor():
    """Create memory monitoring UI"""
    with gr.Accordion("System Monitor", open=False):
        memory_stats = gr.Textbox(
            label="Memory Statistics",
            lines=10,
            interactive=False
        )
        refresh_memory_btn = gr.Button("Refresh Memory Stats")
    
    return memory_stats, refresh_memory_btn