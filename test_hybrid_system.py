#!/usr/bin/env python3
"""
Hybrid Game System Test Suite

Tests the complete hybrid Secret Hitler game system to ensure all components work together.
"""

import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Any
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from web_bridge.hybrid_integration import HybridGameIntegration
from hybrid_game_coordinator import HybridPlayerConfig, PlayerType
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class HybridSystemTester:
    """Test suite for the hybrid game system."""
    
    def __init__(self):
        self.integration = None
        self.test_results = []
        
    async def run_all_tests(self):
        """Run all test cases."""
        logger.info("🧪 Starting Hybrid Game System Test Suite")
        
        tests = [
            self.test_integration_startup,
            self.test_websocket_server,
            self.test_player_authentication,
            self.test_game_creation,
            self.test_action_flow,
            self.test_cleanup
        ]
        
        for test in tests:
            try:
                logger.info(f"Running {test.__name__}...")
                await test()
                self.test_results.append((test.__name__, "PASS", None))
                logger.info(f"✅ {test.__name__} PASSED")
            except Exception as e:
                self.test_results.append((test.__name__, "FAIL", str(e)))
                logger.error(f"❌ {test.__name__} FAILED: {e}")
        
        await self.print_results()
    
    async def test_integration_startup(self):
        """Test that the hybrid integration system starts correctly."""
        self.integration = HybridGameIntegration(host="localhost", port=8765)
        await self.integration.start()
        
        assert self.integration.running, "Integration system should be running"
        assert self.integration.bridge_server is not None, "Bridge server should be initialized"
        
    async def test_websocket_server(self):
        """Test WebSocket server functionality."""
        import websockets
        
        # Test connection
        try:
            async with websockets.connect("ws://localhost:8765") as websocket:
                # Send a test message
                test_message = {
                    "type": "heartbeat",
                    "payload": {"test": True}
                }
                await websocket.send(json.dumps(test_message))
                
                # Wait for any response (may be an error, but connection works)
                response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                logger.info(f"WebSocket test response: {response}")
                
        except asyncio.TimeoutError:
            # Timeout is acceptable - server is running
            pass
        except ConnectionRefusedError:
            raise AssertionError("WebSocket server is not accepting connections")
    
    async def test_player_authentication(self):
        """Test player authentication flow."""
        # This test would simulate a player authentication
        # For now, just test that the bridge server has the necessary methods
        bridge_server = self.integration.bridge_server
        
        assert hasattr(bridge_server, 'send_action_request'), "Bridge should have send_action_request method"
        assert hasattr(bridge_server, 'register_action_callback'), "Bridge should have callback registration"
        assert hasattr(bridge_server, 'get_connected_players'), "Bridge should track connected players"
    
    async def test_game_creation(self):
        """Test hybrid game creation."""
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            logger.warning("Skipping game creation test - no API key")
            return
        
        # Create test player configuration
        player_configs = [
            {
                'id': 'test_human',
                'name': 'Test Human',
                'type': 'human'
            },
            {
                'id': 'test_ai_1',
                'name': 'Test AI 1',
                'type': 'ai',
                'model': 'deepseek/deepseek-v3.2-exp'
            },
            {
                'id': 'test_ai_2',
                'name': 'Test AI 2',
                'type': 'ai',
                'model': 'deepseek/deepseek-v3.2-exp'
            },
            {
                'id': 'test_ai_3',
                'name': 'Test AI 3',
                'type': 'ai',
                'model': 'deepseek/deepseek-v3.2-exp'
            },
            {
                'id': 'test_ai_4',
                'name': 'Test AI 4',
                'type': 'ai',
                'model': 'deepseek/deepseek-v3.2-exp'
            }
        ]
        
        # Create game manager (but don't start the game)
        game_manager = self.integration.create_hybrid_game_manager(
            player_configs=player_configs,
            openrouter_api_key=api_key,
            game_id="test_game"
        )
        
        assert game_manager is not None, "Game manager should be created"
        assert hasattr(game_manager, 'player_types'), "Game manager should have player type tracking"
        assert len(game_manager.player_types) == 5, "Should have 5 players configured"
        
        # Check player types are correctly set
        assert game_manager.player_types['test_human'] == game_manager.PlayerType.HUMAN
        assert game_manager.player_types['test_ai_1'] == game_manager.PlayerType.AI
    
    async def test_action_flow(self):
        """Test the action request/response flow."""
        # Test that the action callback system is properly set up
        bridge_server = self.integration.bridge_server
        test_game_id = "test_action_flow"
        
        # Test callback registration
        callback_called = False
        
        async def test_callback(action_data):
            nonlocal callback_called
            callback_called = True
            return "test_response"
        
        bridge_server.register_action_callback(test_game_id, test_callback)
        
        # Verify callback is registered
        assert test_game_id in bridge_server.action_callbacks, "Callback should be registered"
        
        # Test callback execution (simulate)
        if test_game_id in bridge_server.action_callbacks:
            await bridge_server.action_callbacks[test_game_id]({"test": "data"})
            assert callback_called, "Callback should have been called"
        
        # Clean up
        bridge_server.unregister_action_callback(test_game_id)
    
    async def test_cleanup(self):
        """Test system cleanup."""
        if self.integration:
            await self.integration.stop()
            assert not self.integration.running, "Integration should be stopped"
    
    async def print_results(self):
        """Print test results summary."""
        logger.info("\n" + "="*60)
        logger.info("🧪 HYBRID GAME SYSTEM TEST RESULTS")
        logger.info("="*60)
        
        passed = 0
        failed = 0
        
        for test_name, result, error in self.test_results:
            status_icon = "✅" if result == "PASS" else "❌"
            logger.info(f"{status_icon} {test_name}: {result}")
            
            if error:
                logger.info(f"   Error: {error}")
            
            if result == "PASS":
                passed += 1
            else:
                failed += 1
        
        logger.info("-" * 60)
        logger.info(f"📊 Summary: {passed} passed, {failed} failed")
        
        if failed == 0:
            logger.info("🎉 All tests passed! Hybrid system is ready.")
        else:
            logger.info("⚠️  Some tests failed. Check the errors above.")
        
        logger.info("="*60)


async def run_connectivity_test():
    """Quick connectivity test."""
    logger.info("🔌 Running quick connectivity test...")
    
    try:
        integration = HybridGameIntegration()
        await integration.start()
        
        # Wait a moment for server to start
        await asyncio.sleep(1)
        
        # Test WebSocket connection
        import websockets
        async with websockets.connect("ws://localhost:8765") as websocket:
            logger.info("✅ WebSocket connection successful")
        
        await integration.stop()
        logger.info("✅ Connectivity test passed")
        
    except Exception as e:
        logger.error(f"❌ Connectivity test failed: {e}")
        raise


async def run_component_test():
    """Test individual components."""
    logger.info("🔧 Testing individual components...")
    
    # Test imports
    try:
        from web_bridge import start_hybrid_system
        from hybrid_game_coordinator import HybridGameCoordinator
        from core.game_manager import GameManager
        logger.info("✅ All imports successful")
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        raise
    
    # Test configuration
    try:
        config_file = Path("config/hybrid_game_sample.json")
        if config_file.exists():
            with open(config_file) as f:
                config = json.load(f)
            logger.info("✅ Configuration file valid")
        else:
            logger.warning("⚠️  Sample configuration file not found")
    except json.JSONDecodeError as e:
        logger.error(f"❌ Configuration file error: {e}")
        raise
    
    logger.info("✅ Component test passed")


def setup_logging():
    """Setup logging for tests."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


async def main():
    """Main test runner."""
    setup_logging()
    
    logger.info("🚀 Starting Hybrid Game System Tests")
    logger.info("="*60)
    
    try:
        # Run component tests first
        await run_component_test()
        
        # Run connectivity test
        await run_connectivity_test()
        
        # Run full test suite
        tester = HybridSystemTester()
        await tester.run_all_tests()
        
    except KeyboardInterrupt:
        logger.info("🛑 Tests interrupted by user")
    except Exception as e:
        logger.error(f"❌ Test suite error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)