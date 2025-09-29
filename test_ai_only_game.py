#!/usr/bin/env python3
"""
Enhanced AI-Only Game Test with Detailed Progress Monitoring
This test runs a complete Secret Hitler game with only AI players.
"""
import asyncio
import os
import time
from typing import List, Dict, Any
from dotenv import load_dotenv
from datetime import datetime

from core.game_manager import GameManager

class GameProgressMonitor:
    """Monitor and display game progress in real-time."""
    
    def __init__(self):
        self.start_time = time.time()
        self.phase_times = {}
        self.actions_count = 0
        self.current_phase = None
    
    def log_phase_start(self, phase: str):
        """Log the start of a new phase."""
        now = time.time()
        if self.current_phase:
            self.phase_times[self.current_phase] = now - self.phase_times.get(self.current_phase, now)
        
        self.current_phase = phase
        self.phase_times[phase] = now
        
        elapsed = now - self.start_time
        print(f"\n🎯 [{elapsed:.1f}s] PHASE: {phase.upper()}")
    
    def log_action(self, player_name: str, action_type: str, details: str = ""):
        """Log a player action."""
        self.actions_count += 1
        elapsed = time.time() - self.start_time
        print(f"   [{elapsed:.1f}s] {player_name}: {action_type} {details}")
    
    def log_error(self, error_msg: str):
        """Log an error."""
        elapsed = time.time() - self.start_time
        print(f"❌ [{elapsed:.1f}s] ERROR: {error_msg}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        total_time = time.time() - self.start_time
        return {
            "total_time": total_time,
            "total_actions": self.actions_count,
            "phase_times": dict(self.phase_times),
            "actions_per_second": self.actions_count / total_time if total_time > 0 else 0
        }

async def test_ai_only_game(player_count: int = 5, max_rounds: int = 10):
    """Test a complete AI-only Secret Hitler game."""
    
    # Load environment variables
    load_dotenv()
    
    # Check for API key
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key or api_key == 'your_openrouter_api_key_here':
        print("❌ No valid OpenRouter API key found!")
        print("   1. Get your API key from https://openrouter.ai/keys")
        print("   2. Add it to the .env file: OPENROUTER_API_KEY=your_key_here")
        return False
    
    # Create player configurations
    player_names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry", "Iris", "Jack"]
    player_configs = []
    
    for i in range(player_count):
        player_configs.append({
            "id": f"player{i+1}",
            "name": player_names[i],
            "model": "deepseek/deepseek-v3.2-exp",  # Use free model for testing
            "type": "ai"
        })
    
    # Create progress monitor
    monitor = GameProgressMonitor()
    
    print("🎮 ENHANCED SECRET HITLER AI-ONLY GAME TEST")
    print("=" * 60)
    print(f"Players: {', '.join([p['name'] for p in player_configs])}")
    print(f"Model: Grok-4-Fast (FREE) - Zero cost testing! 🆓")
    print(f"Max Rounds: {max_rounds}")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)
    
    try:
        # Create enhanced game manager with monitoring
        monitor.log_phase_start("INITIALIZATION")
        
        game_manager = EnhancedGameManager(
            player_configs=player_configs,
            openrouter_api_key=api_key,
            game_id=f"ai_test_{int(time.time())}",
            monitor=monitor,
            max_rounds=max_rounds
        )
        
        monitor.log_action("SYSTEM", "Game Manager Created", f"with {player_count} AI players")
        
        # Display initial role assignments
        print("\n🎭 ROLE ASSIGNMENTS:")
        for pid, player in game_manager.game_state.players.items():
            role_emoji = "👑" if player.role.value == "hitler" else "🔴" if player.role.value == "fascist" else "🔵"
            print(f"   {role_emoji} {player.name}: {player.role.value.title()}")
        
        # Start the game
        monitor.log_phase_start("GAME_START")
        monitor.log_action("SYSTEM", "Starting Game", "Beginning main game loop")
        
        result = await game_manager.start_game()
        
        # Check result
        if "error" in result:
            monitor.log_error(f"Game failed: {result['error']}")
            if 'game_state' in result:
                print(f"Game state at failure: {result['game_state']}")
            return False
        
        # Game completed successfully
        monitor.log_phase_start("GAME_COMPLETE")
        
        print("\n" + "=" * 60)
        print("🎉 GAME COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        # Display results
        print(f"🏆 Winner: {result['winner']}")
        print(f"🎯 Win Condition: {result['win_condition']}")
        print(f"⏱️  Duration: {result['duration']:.1f} seconds")
        print(f"💰 Total Cost: ${result['cost_summary']['total_cost']:.4f}")
        print(f"📡 API Requests: {result['cost_summary']['total_requests']}")
        print(f"🚀 Avg Latency: {result['cost_summary']['avg_latency']:.2f}s")
        
        # Display final board state
        final_state = result['final_state']
        print(f"\n📊 FINAL BOARD STATE:")
        print(f"   🔵 Liberal Policies: {final_state['policy_board']['liberal_policies']}/5")
        print(f"   🔴 Fascist Policies: {final_state['policy_board']['fascist_policies']}/6")
        
        # Display performance summary
        perf_summary = monitor.get_summary()
        print(f"\n⚡ PERFORMANCE SUMMARY:")
        print(f"   Total Actions: {perf_summary['total_actions']}")
        print(f"   Actions/Second: {perf_summary['actions_per_second']:.1f}")
        
        # Display logs location
        print(f"\n📁 Detailed logs saved to: logs/{result['game_id']}/")
        
        return True
        
    except Exception as e:
        monitor.log_error(f"Test failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

class EnhancedGameManager(GameManager):
    """Game Manager with enhanced monitoring and debugging."""
    
    def __init__(self, *args, monitor=None, max_rounds=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.monitor = monitor
        self.max_rounds = max_rounds
        self.rounds_completed = 0
    
    async def _execute_game_phase(self):
        """Enhanced phase execution with monitoring."""
        phase = self.game_state.phase
        
        # Check round limit
        if self.rounds_completed >= self.max_rounds:
            if self.monitor:
                self.monitor.log_action("SYSTEM", "Round Limit Reached", f"Stopping at {self.rounds_completed} rounds")
            self.is_running = False
            self.game_state.winner = "TEST_LIMIT"
            self.game_state.win_condition = f"Test stopped after {self.max_rounds} rounds"
            return
        
        # Log phase transitions
        if self.monitor:
            self.monitor.log_phase_start(phase.value)
        
        # Execute the original phase logic
        await super()._execute_game_phase()
        
        # Track rounds
        if phase.value == "NOMINATION":
            self.rounds_completed += 1
            if self.monitor:
                self.monitor.log_action("SYSTEM", "Round Completed", f"Round {self.rounds_completed}/{self.max_rounds}")
    
    async def _make_llm_request(self, player_id: str, prompt: str, decision_type: str):
        """Enhanced LLM request with monitoring."""
        player = self.game_state.players[player_id]
        
        if self.monitor:
            self.monitor.log_action(player.name, f"Making Decision", decision_type)
        
        # Make the request
        try:
            response = await super()._make_llm_request(player_id, prompt, decision_type)
            
            if self.monitor:
                success_str = "✅" if response.success else "❌"
                cost_str = f"${response.cost:.4f}" if response.cost > 0 else "FREE"
                self.monitor.log_action(player.name, f"Response {success_str}", f"{cost_str} - {decision_type}")
                
                # Log the actual response for debugging
                if response.content:
                    # Truncate long responses
                    content_preview = response.content[:100] + "..." if len(response.content) > 100 else response.content
                    print(f"      💭 Response: {content_preview}")
            
            return response
            
        except Exception as e:
            if self.monitor:
                self.monitor.log_error(f"{player.name} request failed: {str(e)}")
            raise

async def main():
    """Main test function."""
    print("AI-Only Secret Hitler Game Test")
    print("Testing game progression with AI agents")
    print()
    
    # Test different configurations
    configs = [
        {"player_count": 5, "max_rounds": 5},  # Quick test
        # {"player_count": 6, "max_rounds": 8},  # Medium test
        # {"player_count": 7, "max_rounds": 10}, # Full test
    ]
    
    for i, config in enumerate(configs, 1):
        print(f"\n🧪 TEST {i}: {config['player_count']} players, {config['max_rounds']} rounds max")
        print("-" * 50)
        
        success = await test_ai_only_game(**config)
        
        if success:
            print(f"✅ Test {i} PASSED")
        else:
            print(f"❌ Test {i} FAILED")
            break
        
        if i < len(configs):
            print("\nWaiting 3 seconds before next test...")
            await asyncio.sleep(3)
    
    print("\n🏁 All tests completed!")

if __name__ == "__main__":
    asyncio.run(main())