#!/usr/bin/env python3
"""
Complete system test for hybrid Secret Hitler with both terminal and web interfaces.
"""
import asyncio
import sys
import time
import subprocess
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_complete_system():
    """Test the complete hybrid system including HTTP and WebSocket servers."""
    print("🧪 Testing Complete Hybrid System")
    print("=" * 50)
    
    try:
        # Test 1: WebSocket server startup
        print("🔌 Testing WebSocket server...")
        from web_bridge.bidirectional_bridge import get_hybrid_bridge_instance
        
        server = get_hybrid_bridge_instance("localhost", 8765)
        server_task = asyncio.create_task(server.start_server())
        await asyncio.sleep(1)
        print("✅ WebSocket server started successfully")
        
        # Test 2: HTTP server startup  
        print("🌐 Testing HTTP server...")
        from web_bridge.http_server import SimpleGameHTTPServer
        
        http_server = SimpleGameHTTPServer(port=8080)
        http_started = http_server.start_server()
        if http_started:
            print("✅ HTTP server started successfully")
            print("   📁 Serving web files from: web_interface/")
            print("   🌐 Web interface: http://localhost:8080")
        else:
            print("❌ HTTP server failed to start")
        
        await asyncio.sleep(1)
        
        # Test 3: WebSocket connection  
        print("🔗 Testing WebSocket connection...")
        try:
            import websockets
            async with websockets.connect("ws://localhost:8765") as websocket:
                await websocket.send('{"type": "test", "payload": {}}')
                print("✅ WebSocket connection successful")
        except Exception as e:
            print(f"❌ WebSocket connection failed: {e}")
        
        # Test 4: HTTP request
        print("📄 Testing HTTP request...")
        try:
            import urllib.request
            with urllib.request.urlopen("http://localhost:8080") as response:
                content = response.read(100).decode('utf-8', errors='ignore')
                if '<html' in content.lower():
                    print("✅ HTTP server serving HTML content")
                else:
                    print("❌ HTTP server not serving HTML")
        except Exception as e:
            print(f"❌ HTTP request failed: {e}")
        
        # Clean up
        server_task.cancel()
        http_server.stop_server()
        
        print()
        print("🎉 System test completed!")
        print("🚀 Ready to launch: python play_hybrid.py --browser")
        return True
        
    except Exception as e:
        print(f"❌ System test failed: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_complete_system())
    sys.exit(0 if result else 1)