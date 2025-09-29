"""
Simple HTTP server to serve the web interface for Secret Hitler hybrid games.
Integrates with the WebSocket server to provide a complete web experience.
"""

import asyncio
import logging
import mimetypes
from pathlib import Path
from typing import Optional
import http.server
import socketserver
import threading

logger = logging.getLogger(__name__)

class SimpleGameHTTPServer:
    """Simple HTTP server for serving game web interface."""
    
    def __init__(self, port: int = 8080, web_root: Optional[Path] = None):
        self.port = port
        self.web_root = web_root or (Path(__file__).parent.parent / "web_interface")
        self.server = None
        self.server_thread = None
        
    def start_server(self):
        """Start the HTTP server in a background thread."""
        if self.server_thread and self.server_thread.is_alive():
            logger.warning("HTTP server already running")
            return
        
        # Make sure web root exists
        if not self.web_root.exists():
            logger.error(f"Web root directory does not exist: {self.web_root}")
            return False
        
        self.server_thread = threading.Thread(target=self._run_server, daemon=True)
        self.server_thread.start()
        
        logger.info(f"HTTP server started on port {self.port}, serving {self.web_root}")
        return True
    
    def _run_server(self):
        """Run the HTTP server."""
        try:
            handler = self._create_handler()
            with socketserver.TCPServer(("", self.port), handler) as httpd:
                self.server = httpd
                logger.info(f"Serving at http://localhost:{self.port}")
                httpd.serve_forever()
        except Exception as e:
            logger.error(f"HTTP server error: {e}")
    
    def _create_handler(self):
        """Create a custom HTTP handler for serving files."""
        web_root = self.web_root
        
        class GameHTTPHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(web_root), **kwargs)
            
            def do_GET(self):
                """Handle GET requests."""
                # Default to game.html for root path
                if self.path == '/' or self.path == '/game':
                    self.path = '/game.html'
                
                # Log the request
                logger.debug(f"HTTP request: {self.path}")
                
                super().do_GET()
            
            def log_message(self, format, *args):
                """Override to use our logger."""
                logger.debug(f"HTTP {self.address_string()} - {format % args}")
        
        return GameHTTPHandler
    
    def stop_server(self):
        """Stop the HTTP server."""
        if self.server:
            self.server.shutdown()
            self.server = None
        
        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout=5.0)
        
        logger.info("HTTP server stopped")

# Integration with hybrid system
class IntegratedWebServer:
    """Integrated web server that serves both WebSocket and HTTP."""
    
    def __init__(self, ws_port: int = 8765, http_port: int = 8080):
        self.ws_port = ws_port
        self.http_port = http_port
        self.http_server = SimpleGameHTTPServer(http_port)
        self.websocket_server = None
    
    async def start_integrated_server(self, websocket_server):
        """Start both HTTP and WebSocket servers."""
        self.websocket_server = websocket_server
        
        # Start HTTP server
        if not self.http_server.start_server():
            logger.error("Failed to start HTTP server")
            return False
        
        logger.info(f"Integrated web server started:")
        logger.info(f"  🌐 Web interface: http://localhost:{self.http_port}")
        logger.info(f"  🔌 WebSocket: ws://localhost:{self.ws_port}")
        
        return True
    
    async def stop_integrated_server(self):
        """Stop both servers."""
        self.http_server.stop_server()
        logger.info("Integrated web server stopped")

# Utility function to create a complete web interface
def create_complete_web_interface(game_id: str, websocket_port: int = 8765, http_port: int = 8080):
    """Create a complete web interface with both HTTP and WebSocket servers."""
    
    # Update the HTML file to include the game ID if needed
    web_root = Path(__file__).parent.parent / "web_interface"
    html_file = web_root / "game.html"
    
    if html_file.exists():
        # Could customize HTML content here if needed
        pass
    
    integrated_server = IntegratedWebServer(websocket_port, http_port)
    return integrated_server

if __name__ == "__main__":
    # Test the HTTP server standalone
    logging.basicConfig(level=logging.INFO)
    
    server = SimpleGameHTTPServer()
    server.start_server()
    
    try:
        # Keep the main thread alive
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping server...")
        server.stop_server()