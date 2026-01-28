#!/usr/bin/env python3
"""
Main entry point for Stable Diffusion WebUI
Optimized for Google Colab
"""

import argparse
import torch
from app import Config, TunnelManager, create_interface

def check_gpu():
    """Check if GPU is available"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ GPU Detected: {gpu_name}")
        print(f"✓ GPU Memory: {gpu_memory:.2f} GB")
        return True
    else:
        print("⚠ No GPU detected. Running on CPU (will be slow)")
        return False

def main():
    parser = argparse.ArgumentParser(description="Stable Diffusion WebUI")
    parser.add_argument("--share", action="store_true", help="Create public URL")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    parser.add_argument("--server-name", type=str, default="0.0.0.0", help="Server name")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("STABLE DIFFUSION WEBUI - GOOGLE COLAB OPTIMIZED")
    print("="*60 + "\n")
    
    # Load configuration
    print("📋 Loading configuration...")
    config = Config(args.config)
    config.create_directories()
    print("✓ Configuration loaded\n")
    
    # Check GPU
    print("🔍 Checking hardware...")
    has_gpu = check_gpu()
    print()
    
    # Start tunnel if enabled
    tunnel_manager = None
    if args.share or config.get('tunnel.enabled', False):
        print("🌐 Starting tunnel...")
        tunnel_manager = TunnelManager(config)
        try:
            tunnel_manager.start_ngrok(args.port)
        except Exception as e:
            print(f"⚠ Could not start tunnel: {e}")
            print("Continuing without tunnel...\n")
    
    # Launch UI
    print("🚀 Launching WebUI...")
    print(f"Local URL: http://localhost:{args.port}")
    print()
    
    demo = create_interface(config)
    demo.queue(max_size=20)
    
    try:
        demo.launch(
            server_name=args.server_name,
            server_port=args.port,
            share=args.share,
            inbrowser=True,
            show_error=True
        )
    except KeyboardInterrupt:
        print("\n\n✓ Shutting down...")
        if tunnel_manager:
            tunnel_manager.stop()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        if tunnel_manager:
            tunnel_manager.stop()

if __name__ == "__main__":
    main()
