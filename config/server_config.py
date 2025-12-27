#!/usr/bin/env python3
"""
Server Configuration for WeSee Application
"""

import os
import sys
import warnings
import logging

class ServerConfig:
    """Simple Flask server configuration"""
    
    def start_server(self, app):
        """Start Flask server with suppressed warnings"""
        # Suppress all warnings completely
        warnings.filterwarnings("ignore")
        logging.getLogger('werkzeug').disabled = True
        
        # Hide Flask startup messages completely
        import flask.cli
        flask.cli.show_server_banner = lambda *args: None
        
        # Clean startup message
        print("🚀 WeSee Server Starting...")
        print("📱 Access at: http://127.0.0.1:5000")
        print("⏹️  Press Ctrl+C to stop")
        print("-" * 50)
        
        # Redirect stderr before starting Flask
        original_stderr = sys.stderr
        with open(os.devnull, 'w') as devnull:
            sys.stderr = devnull
            try:
                # Start server (Flask messages will be hidden)
                app.run(debug=False, host='127.0.0.1', port=5000, use_reloader=False)
            finally:
                sys.stderr = original_stderr

# Global instance
server_config = ServerConfig()
