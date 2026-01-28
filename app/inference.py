import torch
import numpy as np
from PIL import Image
from typing import Optional, List, Union
import io
import base64

class InferenceEngine:
    """Handles image generation inference"""
    
    def __init__(self, model_loader):
        self.model_loader = model_loader
        self.config = model_loader.config
        
    def prepare_image(self, image, target_size=None):
        """Prepare image for processing"""
        if isinstance(image, str):
            # Base64 or file path
            if image.startswith('data:image'):
                # Base64
                image_data = image.split(',')[1]
                image = Image.open(io.BytesIO(base64.b64decode(image_data)))
            else:
                # File path
                image = Image.open(image)
        
        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if needed
        if target_size:
            image = image.resize(target_size, Image.LANCZOS)
        
        return image
    
    def generate_txt2img(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        num_images: int = 1,
        seed: Optional[int] = None,
        **kwargs
    ):
        """
        Generate images from text prompt
        
        Args:
            prompt: Text prompt
            negative_prompt: Negative prompt
            width: Image width
            height: Image height
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale
            num_images: Number of images to generate
            seed: Random seed for reproducibility
        """
        if not self.model_loader.current_pipeline:
            raise RuntimeError("No model loaded. Please load a model first.")
        
        pipe = self.model_loader.current_pipeline
        
        # Set seed for reproducibility
        if seed is not None:
            generator = torch.Generator(device=self.model_loader.device).manual_seed(seed)
        else:
            generator = None
        
        print(f"\nGenerating {num_images} image(s)...")
        print(f"Prompt: {prompt[:100]}...")
        print(f"Size: {width}x{height}")
        print(f"Steps: {num_inference_steps}")
        print(f"Guidance: {guidance_scale}")
        if seed is not None:
            print(f"Seed: {seed}")
        
        try:
            # Generate
            with torch.inference_mode():
                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=num_images,
                    generator=generator,
                    **kwargs
                )
            
            images = result.images
            
            print(f"✓ Generated {len(images)} image(s) successfully\n")
            
            return images
            
        except Exception as e:
            print(f"✗ Error during generation: {e}")
            raise
    
    def generate_img2img(
        self,
        prompt: str,
        image: Union[Image.Image, str],
        negative_prompt: str = "",
        strength: float = 0.75,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        num_images: int = 1,
        seed: Optional[int] = None,
        **kwargs
    ):
        """
        Generate images from image + text prompt
        
        Args:
            prompt: Text prompt
            image: Input image
            negative_prompt: Negative prompt
            strength: How much to transform the image (0-1)
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale
            num_images: Number of images to generate
            seed: Random seed
        """
        if not self.model_loader.current_pipeline:
            raise RuntimeError("No model loaded. Please load a model first.")
        
        # Prepare image
        init_image = self.prepare_image(image)
        
        # Set seed
        if seed is not None:
            generator = torch.Generator(device=self.model_loader.device).manual_seed(seed)
        else:
            generator = None
        
        print(f"\nGenerating {num_images} image(s) from image...")
        print(f"Prompt: {prompt[:100]}...")
        print(f"Strength: {strength}")
        print(f"Steps: {num_inference_steps}")
        
        try:
            # For img2img, we need to check if pipeline supports it
            # If using base pipeline, we need to switch to img2img pipeline
            from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionXLImg2ImgPipeline
            
            # Create img2img pipeline from existing components
            if self.model_loader.model_type == 'sdxl':
                img2img_pipe = StableDiffusionXLImg2ImgPipeline(
                    vae=self.model_loader.current_pipeline.vae,
                    text_encoder=self.model_loader.current_pipeline.text_encoder,
                    text_encoder_2=self.model_loader.current_pipeline.text_encoder_2,
                    tokenizer=self.model_loader.current_pipeline.tokenizer,
                    tokenizer_2=self.model_loader.current_pipeline.tokenizer_2,
                    unet=self.model_loader.current_pipeline.unet,
                    scheduler=self.model_loader.current_pipeline.scheduler,
                ).to(self.model_loader.device)
            else:
                img2img_pipe = StableDiffusionImg2ImgPipeline(
                    vae=self.model_loader.current_pipeline.vae,
                    text_encoder=self.model_loader.current_pipeline.text_encoder,
                    tokenizer=self.model_loader.current_pipeline.tokenizer,
                    unet=self.model_loader.current_pipeline.unet,
                    scheduler=self.model_loader.current_pipeline.scheduler,
                    safety_checker=None,
                    feature_extractor=None,
                ).to(self.model_loader.device)
            
            # Apply same memory optimizations
            img2img_pipe = self.model_loader.memory_manager.optimize_pipeline(img2img_pipe)
            
            with torch.inference_mode():
                result = img2img_pipe(
                    prompt=prompt,
                    image=init_image,
                    negative_prompt=negative_prompt,
                    strength=strength,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=num_images,
                    generator=generator,
                    **kwargs
                )
            
            images = result.images
            
            print(f"✓ Generated {len(images)} image(s) successfully\n")
            
            return images
            
        except Exception as e:
            print(f"✗ Error during generation: {e}")
            raise
    
    def save_images(self, images, output_dir="outputs", prefix="image"):
        """Save generated images to disk"""
        import os
        from datetime import datetime
        
        os.makedirs(output_dir, exist_ok=True)
        
        saved_paths = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for i, image in enumerate(images):
            filename = f"{prefix}_{timestamp}_{i:04d}.png"
            filepath = os.path.join(output_dir, filename)
            image.save(filepath)
            saved_paths.append(filepath)
            print(f"✓ Saved: {filepath}")
        
        return saved_paths