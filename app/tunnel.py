import os
from pyngrok import ngrok, conf

class TunnelManager:
    def __init__(self, config):
        self.config = config
        self.tunnel = None
        
    def start_ngrok(self, port=7860):
        """Start ngrok tunnel"""
        token = self.config.get('tunnel.ngrok_token')
        
        if token:
            ngrok.set_auth_token(token)
        
        # Start tunnel
        self.tunnel = ngrok.connect(port, bind_tls=True)
        public_url = self.tunnel.public_url
        
        print(f"\n{'='*60}")
        print(f"✓ Ngrok Tunnel Started!")
        print(f"✓ Public URL: {public_url}")
        print(f"{'='*60}\n")
        
        return public_url
    
    def stop(self):
        """Stop tunnel"""
        if self.tunnel:
            ngrok.disconnect(self.tunnel.public_url)
            print("✓ Tunnel stopped")
    
    def __del__(self):
        """Cleanup on deletion"""
        self.stop()