#!/usr/bin/env python3
"""
Hybrid Game Setup Script

This script helps you set up and test the hybrid Secret Hitler game system.
It checks dependencies, validates configuration, and provides quick start options.
"""

import asyncio
import json
import os
import sys
import subprocess
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class HybridGameSetup:
    """Setup and validation for hybrid games."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.env_file = self.project_root / ".env"
        self.config_dir = self.project_root / "config"
        self.requirements_file = self.project_root / "requirements.txt"
        
    def check_python_version(self):
        """Check Python version compatibility."""
        logger.info("🐍 Checking Python version...")
        
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            raise RuntimeError(f"Python 3.8+ required, found {version.major}.{version.minor}")
        
        logger.info(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    
    def check_dependencies(self):
        """Check if required Python packages are installed."""
        logger.info("📦 Checking dependencies...")
        
        required_packages = [
            'websockets',
            'asyncio',
            'dotenv',
            'openai',
            'requests',
            'pydantic'
        ]
        
        missing = []
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
                logger.info(f"✅ {package}")
            except ImportError:
                missing.append(package)
                logger.warning(f"❌ {package} (missing)")
        
        if missing:
            logger.info("\n📥 Installing missing dependencies...")
            self.install_dependencies()
        else:
            logger.info("✅ All dependencies satisfied")
    
    def install_dependencies(self):
        """Install required dependencies."""
        if not self.requirements_file.exists():
            logger.error("❌ requirements.txt not found")
            return False
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", str(self.requirements_file)
            ])
            logger.info("✅ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install dependencies: {e}")
            return False
    
    def check_environment(self):
        """Check environment configuration."""
        logger.info("🔧 Checking environment configuration...")
        
        if not self.env_file.exists():
            logger.info("📝 Creating .env file from template...")
            self.create_env_file()
        
        # Load and validate .env
        from dotenv import load_dotenv
        load_dotenv(self.env_file)
        
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key or api_key == 'your_api_key_here':
            logger.warning("⚠️  OPENROUTER_API_KEY not set or is placeholder")
            logger.info("   You'll need to add your OpenRouter API key to .env")
            logger.info("   Get one from: https://openrouter.ai/")
        else:
            logger.info("✅ OPENROUTER_API_KEY configured")
        
        default_model = os.getenv('DEFAULT_MODEL', 'deepseek/deepseek-v3.2-exp')
        logger.info(f"✅ Default AI model: {default_model}")
    
    def create_env_file(self):
        """Create .env file with default values."""
        env_content = """# OpenRouter API Configuration
OPENROUTER_API_KEY=your_api_key_here

# Default AI Model (free tier recommended for testing)
DEFAULT_MODEL=deepseek/deepseek-v3.2-exp

# WebSocket Server Configuration
WEBSOCKET_HOST=localhost
WEBSOCKET_PORT=8765

# Game Configuration
HUMAN_ACTION_TIMEOUT=60.0
LOG_LEVEL=INFO
"""
        
        with open(self.env_file, 'w') as f:
            f.write(env_content)
        
        logger.info(f"✅ Created {self.env_file}")
    
    def check_configuration(self):
        """Check configuration files."""
        logger.info("📋 Checking configuration files...")
        
        # Ensure config directory exists
        self.config_dir.mkdir(exist_ok=True)
        
        sample_config = self.config_dir / "hybrid_game_sample.json"
        if sample_config.exists():
            try:
                with open(sample_config) as f:
                    config = json.load(f)
                logger.info("✅ Sample configuration valid")
            except json.JSONDecodeError as e:
                logger.error(f"❌ Sample configuration invalid: {e}")
        else:
            logger.info("📝 Sample configuration file exists")
    
    def check_project_structure(self):
        """Verify project structure."""
        logger.info("📂 Checking project structure...")
        
        required_files = [
            "hybrid_game_launcher.py",
            "hybrid_game_coordinator.py",
            "hybrid_player_client.html",
            "core/game_manager.py",
            "web_bridge/__init__.py",
            "web_bridge/bidirectional_bridge.py",
            "web_bridge/hybrid_integration.py"
        ]
        
        missing_files = []
        for file_path in required_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                logger.info(f"✅ {file_path}")
            else:
                missing_files.append(file_path)
                logger.error(f"❌ {file_path} (missing)")
        
        if missing_files:
            logger.error("❌ Some required files are missing")
            return False
        
        logger.info("✅ Project structure complete")
        return True
    
    async def test_system(self):
        """Run system tests."""
        logger.info("🧪 Running system tests...")
        
        try:
            # Import test module
            test_script = self.project_root / "test_hybrid_system.py"
            if test_script.exists():
                # Run the test script
                result = subprocess.run([
                    sys.executable, str(test_script)
                ], capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    logger.info("✅ System tests passed")
                    return True
                else:
                    logger.error(f"❌ System tests failed:\n{result.stderr}")
                    return False
            else:
                logger.warning("⚠️  Test script not found, skipping tests")
                return True
                
        except subprocess.TimeoutExpired:
            logger.error("❌ System tests timed out")
            return False
        except Exception as e:
            logger.error(f"❌ Error running tests: {e}")
            return False
    
    def print_usage_instructions(self):
        """Print usage instructions."""
        logger.info("\n" + "="*60)
        logger.info("🎮 HYBRID GAME SYSTEM READY!")
        logger.info("="*60)
        logger.info("""
🚀 Quick Start Options:

1. Launch with 1 human + 4 AI players:
   python hybrid_game_launcher.py --quick-start --humans 1 --ais 4

2. Interactive setup:
   python hybrid_game_launcher.py --interactive

3. Use configuration file:
   python hybrid_game_launcher.py --config config/hybrid_game_sample.json

🌐 Human Player Connection:
   - Open hybrid_player_client.html in your browser
   - Enter the Player ID and Game ID from the launcher
   - Click "Connect" then "Authenticate"

📚 Documentation:
   - Read README_HYBRID_GAMES.md for detailed instructions
   - Check logs/[game_id]/ for game logs and debugging

🔧 Configuration:
   - Edit .env for API keys and settings
   - Modify config/*.json for custom game setups

🧪 Testing:
   python test_hybrid_system.py
""")
        logger.info("="*60)
    
    async def run_setup(self):
        """Run complete setup process."""
        logger.info("🛠️  Setting up Hybrid Secret Hitler Game System")
        logger.info("="*60)
        
        try:
            # Core checks
            self.check_python_version()
            self.check_dependencies()
            self.check_environment()
            self.check_configuration()
            
            # Project structure
            if not self.check_project_structure():
                logger.error("❌ Setup failed: missing required files")
                return False
            
            # System tests
            test_passed = await self.test_system()
            
            if test_passed:
                logger.info("✅ Setup completed successfully!")
                self.print_usage_instructions()
                return True
            else:
                logger.warning("⚠️  Setup completed with test failures")
                logger.info("   System may still work, but check the errors above")
                self.print_usage_instructions()
                return True
                
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def setup_logging():
    """Setup logging for setup script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


async def main():
    """Main setup function."""
    setup_logging()
    
    setup = HybridGameSetup()
    success = await setup.run_setup()
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)