"""
Batch experiment runner for Secret Hitler LLM evaluation.
Runs multiple games in parallel and collects comprehensive statistics.
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import logging
from concurrent.futures import ProcessPoolExecutor
import argparse

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from core.game_manager import GameManager
from web_bridge import create_adapter, start_websocket_server

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExperimentConfig:
    """Configuration for batch experiments."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self.name = config_dict.get('name', 'unnamed_experiment')
        self.description = config_dict.get('description', '')
        self.num_games = config_dict.get('num_games', 10)
        self.max_parallel = config_dict.get('max_parallel', 3)
        self.player_configs = config_dict.get('player_configs', [])
        self.game_settings = config_dict.get('game_settings', {})
        self.cost_limit_per_game = config_dict.get('cost_limit_per_game', 5.0)
        self.total_cost_limit = config_dict.get('total_cost_limit', 50.0)
        self.enable_web_monitoring = config_dict.get('enable_web_monitoring', False)
        self.output_dir = config_dict.get('output_dir', 'experiment_results')
        self.models_to_test = config_dict.get('models_to_test', ['grok-4-fast-free'])
        
    def validate(self) -> bool:
        """Validate the experiment configuration."""
        if not self.player_configs:
            logger.error("No player configurations provided")
            return False
        
        if self.num_games <= 0:
            logger.error("Number of games must be positive")
            return False
        
        if len(self.player_configs) < 2:
            logger.error("At least 2 players required")
            return False
        
        return True

class BatchRunner:
    """Runs batch experiments with multiple Secret Hitler games."""
    
    def __init__(self, config: ExperimentConfig, api_key: str):
        self.config = config
        self.api_key = api_key
        self.experiment_id = f"{config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir = Path(config.output_dir) / self.experiment_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Experiment tracking
        self.completed_games = 0
        self.failed_games = 0
        self.total_cost = 0.0
        self.game_results = []
        self.start_time = None
        self.end_time = None
        
        # Web monitoring
        self.websocket_server_task = None
        
    async def run_experiment(self) -> Dict[str, Any]:
        """Run the complete batch experiment."""
        if not self.config.validate():
            raise ValueError("Invalid experiment configuration")
        
        logger.info(f"Starting experiment: {self.config.name}")
        logger.info(f"Games to run: {self.config.num_games}")
        logger.info(f"Max parallel: {self.config.max_parallel}")
        logger.info(f"Output directory: {self.output_dir}")
        
        self.start_time = datetime.now()
        
        try:
            # Start web monitoring if enabled
            if self.config.enable_web_monitoring:
                await self._start_web_monitoring()
            
            # Save experiment configuration
            await self._save_experiment_config()
            
            # Run games in batches
            await self._run_games_in_batches()
            
            # Generate final report
            final_report = await self._generate_final_report()
            
            self.end_time = datetime.now()
            
            logger.info(f"Experiment completed successfully!")
            logger.info(f"Games completed: {self.completed_games}/{self.config.num_games}")
            logger.info(f"Games failed: {self.failed_games}")
            logger.info(f"Total cost: ${self.total_cost:.4f}")
            logger.info(f"Duration: {(self.end_time - self.start_time).total_seconds():.1f}s")
            
            return final_report
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            raise
        finally:
            # Stop web monitoring
            if self.websocket_server_task:
                self.websocket_server_task.cancel()
    
    async def _start_web_monitoring(self):
        """Start WebSocket server for web monitoring."""
        try:
            self.websocket_server_task = asyncio.create_task(
                start_websocket_server(port=8766)  # Different port for experiments
            )
            await asyncio.sleep(1)  # Give server time to start
            logger.info("Web monitoring available at ws://localhost:8766")
        except Exception as e:
            logger.warning(f"Failed to start web monitoring: {e}")
    
    async def _save_experiment_config(self):
        """Save experiment configuration to file."""
        config_file = self.output_dir / "experiment_config.json"
        config_data = {
            'experiment_id': self.experiment_id,
            'config': self.config.__dict__,
            'start_time': self.start_time.isoformat(),
            'python_version': sys.version,
            'working_directory': str(Path.cwd())
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    async def _run_games_in_batches(self):
        """Run games in parallel batches."""
        game_tasks = []
        
        for i in range(self.config.num_games):
            if self.total_cost >= self.config.total_cost_limit:
                logger.warning(f"Total cost limit reached: ${self.total_cost:.4f}")
                break
            
            game_id = f"{self.experiment_id}_game_{i+1:03d}"
            task = self._run_single_game(game_id, i+1)
            game_tasks.append(task)
            
            # Run in batches to avoid overwhelming the API
            if len(game_tasks) >= self.config.max_parallel or i == self.config.num_games - 1:
                results = await asyncio.gather(*game_tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, Exception):
                        logger.error(f"Game failed with exception: {result}")
                        self.failed_games += 1
                    else:
                        self._process_game_result(result)
                
                game_tasks = []
                
                # Brief pause between batches
                await asyncio.sleep(2)
    
    async def _run_single_game(self, game_id: str, game_number: int) -> Dict[str, Any]:
        """Run a single game and return results."""
        logger.info(f"Starting game {game_number}: {game_id}")
        
        try:
            # Create game manager
            game_manager = GameManager(
                player_configs=self.config.player_configs.copy(),
                openrouter_api_key=self.api_key,
                game_id=game_id
            )
            
            # Set up web monitoring if enabled
            if self.config.enable_web_monitoring:
                adapter = create_adapter(game_manager)
                await adapter.start_monitoring()
            
            # Run the game
            result = await game_manager.start_game()
            
            if "error" in result:
                logger.error(f"Game {game_id} failed: {result['error']}")
                return {
                    'game_id': game_id,
                    'game_number': game_number,
                    'success': False,
                    'error': result['error'],
                    'cost': 0.0
                }
            
            # Extract game data
            cost = result.get('cost_summary', {}).get('total_cost', 0.0)
            
            logger.info(f"Game {game_number} completed. Cost: ${cost:.4f}")
            
            return {
                'game_id': game_id,
                'game_number': game_number,
                'success': True,
                'result': result,
                'cost': cost,
                'duration': result.get('duration', 0),
                'winner': result.get('winner'),
                'win_condition': result.get('win_condition'),
                'final_state': result.get('final_state', {}),
                'cost_summary': result.get('cost_summary', {}),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Game {game_id} crashed: {e}")
            return {
                'game_id': game_id,
                'game_number': game_number,
                'success': False,
                'error': str(e),
                'cost': 0.0,
                'timestamp': datetime.now().isoformat()
            }
    
    def _process_game_result(self, result: Dict[str, Any]):
        """Process and store a game result."""
        if result['success']:
            self.completed_games += 1
            self.total_cost += result['cost']
            
            # Save individual game result
            game_file = self.output_dir / f"game_{result['game_number']:03d}.json"
            with open(game_file, 'w') as f:
                json.dump(result, f, indent=2)
        else:
            self.failed_games += 1
        
        self.game_results.append(result)
        
        # Save progress
        self._save_progress()
    
    def _save_progress(self):
        """Save current experiment progress."""
        progress_file = self.output_dir / "progress.json"
        progress_data = {
            'experiment_id': self.experiment_id,
            'completed_games': self.completed_games,
            'failed_games': self.failed_games,
            'total_cost': self.total_cost,
            'games_remaining': self.config.num_games - self.completed_games - self.failed_games,
            'last_updated': datetime.now().isoformat()
        }
        
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
    
    async def _generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final report."""
        successful_games = [r for r in self.game_results if r['success']]
        
        # Calculate statistics
        if successful_games:
            avg_cost = sum(g['cost'] for g in successful_games) / len(successful_games)
            avg_duration = sum(g['duration'] for g in successful_games) / len(successful_games)
            
            # Win rate analysis
            liberal_wins = sum(1 for g in successful_games if g['winner'] == 'liberal')
            fascist_wins = sum(1 for g in successful_games if g['winner'] == 'fascist')
            
            # Win condition analysis
            win_conditions = {}
            for game in successful_games:
                condition = game['win_condition']
                win_conditions[condition] = win_conditions.get(condition, 0) + 1
            
            # Cost efficiency
            total_actions = sum(
                g.get('final_state', {}).get('total_actions', 0) 
                for g in successful_games
            )
            cost_per_action = self.total_cost / max(total_actions, 1)
        else:
            avg_cost = avg_duration = 0
            liberal_wins = fascist_wins = 0
            win_conditions = {}
            cost_per_action = 0
        
        report = {
            'experiment_summary': {
                'experiment_id': self.experiment_id,
                'name': self.config.name,
                'description': self.config.description,
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'end_time': self.end_time.isoformat() if self.end_time else None,
                'duration_seconds': (self.end_time - self.start_time).total_seconds() if self.start_time and self.end_time else 0
            },
            'game_statistics': {
                'total_games_attempted': self.config.num_games,
                'successful_games': self.completed_games,
                'failed_games': self.failed_games,
                'success_rate': self.completed_games / max(self.config.num_games, 1),
                'average_game_cost': avg_cost,
                'average_game_duration': avg_duration,
                'total_cost': self.total_cost,
                'cost_per_action': cost_per_action
            },
            'gameplay_analysis': {
                'liberal_wins': liberal_wins,
                'fascist_wins': fascist_wins,
                'liberal_win_rate': liberal_wins / max(len(successful_games), 1),
                'fascist_win_rate': fascist_wins / max(len(successful_games), 1),
                'win_conditions': win_conditions
            },
            'cost_analysis': {
                'total_cost': self.total_cost,
                'cost_per_game': self.total_cost / max(self.completed_games, 1),
                'cost_efficiency': {
                    'cost_per_action': cost_per_action,
                    'games_per_dollar': self.completed_games / max(self.total_cost, 0.01)
                }
            },
            'configuration': self.config.__dict__,
            'raw_results': self.game_results
        }
        
        # Save report
        report_file = self.output_dir / "final_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate summary text
        await self._generate_summary_text(report)
        
        return report
    
    async def _generate_summary_text(self, report: Dict[str, Any]):
        """Generate human-readable summary."""
        summary_file = self.output_dir / "summary.txt"
        
        stats = report['game_statistics']
        gameplay = report['gameplay_analysis']
        cost = report['cost_analysis']
        
        summary = f"""
Secret Hitler LLM Experiment Summary
=====================================

Experiment: {self.config.name}
ID: {self.experiment_id}
Description: {self.config.description}

Game Results:
- Games Attempted: {stats['total_games_attempted']}
- Games Completed: {stats['successful_games']}
- Games Failed: {stats['failed_games']}
- Success Rate: {stats['success_rate']:.1%}

Gameplay Analysis:
- Liberal Wins: {gameplay['liberal_wins']} ({gameplay['liberal_win_rate']:.1%})
- Fascist Wins: {gameplay['fascist_wins']} ({gameplay['fascist_win_rate']:.1%})

Win Conditions:
"""
        
        for condition, count in gameplay['win_conditions'].items():
            summary += f"- {condition}: {count}\n"
        
        summary += f"""
Cost Analysis:
- Total Cost: ${cost['total_cost']:.4f}
- Average Cost per Game: ${cost['cost_per_game']:.4f}
- Cost per Action: ${cost['cost_efficiency']['cost_per_action']:.4f}
- Games per Dollar: {cost['cost_efficiency']['games_per_dollar']:.1f}

Performance:
- Average Game Duration: {stats['average_game_duration']:.1f} seconds
- Total Experiment Duration: {report['experiment_summary']['duration_seconds']:.1f} seconds

Files Generated:
- Individual game logs: logs/[game_id]/
- Game results: game_001.json, game_002.json, ...
- Final report: final_report.json
- Experiment config: experiment_config.json
"""
        
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        logger.info(f"Summary saved to: {summary_file}")

def create_experiment_config(template: str = "default") -> Dict[str, Any]:
    """Create experiment configuration templates."""
    
    if template == "quick_test":
        return {
            "name": "quick_test",
            "description": "Quick test with 2 games for validation",
            "num_games": 2,
            "max_parallel": 1,
            "player_configs": [
                {"id": "player1", "name": "Alice", "model": "grok-4-fast-free"},
                {"id": "player2", "name": "Bob", "model": "grok-4-fast-free"}
            ],
            "cost_limit_per_game": 0.01,
            "total_cost_limit": 0.02,
            "enable_web_monitoring": False,
            "output_dir": "experiment_results"
        }
    
    elif template == "model_comparison":
        return {
            "name": "model_comparison",
            "description": "Compare different models in 5-player games",
            "num_games": 20,
            "max_parallel": 2,
            "player_configs": [
                {"id": "player1", "name": "Alice", "model": "grok-4-fast-free"},
                {"id": "player2", "name": "Bob", "model": "grok-4-fast-free"},
                {"id": "player3", "name": "Charlie", "model": "grok-4-fast-free"},
                {"id": "player4", "name": "Diana", "model": "grok-4-fast-free"},
                {"id": "player5", "name": "Eve", "model": "grok-4-fast-free"}
            ],
            "cost_limit_per_game": 1.0,
            "total_cost_limit": 25.0,
            "enable_web_monitoring": True,
            "output_dir": "experiment_results"
        }
    
    else:  # default
        return {
            "name": "baseline_experiment",
            "description": "Baseline Secret Hitler LLM experiment",
            "num_games": 10,
            "max_parallel": 3,
            "player_configs": [
                {"id": "player1", "name": "Alice", "model": "grok-4-fast-free"},
                {"id": "player2", "name": "Bob", "model": "grok-4-fast-free"},
                {"id": "player3", "name": "Charlie", "model": "grok-4-fast-free"},
                {"id": "player4", "name": "Diana", "model": "grok-4-fast-free"},
                {"id": "player5", "name": "Eve", "model": "grok-4-fast-free"}
            ],
            "cost_limit_per_game": 2.0,
            "total_cost_limit": 25.0,
            "enable_web_monitoring": False,
            "output_dir": "experiment_results"
        }

async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Secret Hitler LLM Batch Experiment Runner")
    parser.add_argument('--config', type=str, help='Path to experiment config JSON file')
    parser.add_argument('--template', type=str, choices=['default', 'quick_test', 'model_comparison'], 
                       default='default', help='Use a predefined experiment template')
    parser.add_argument('--api-key', type=str, help='OpenRouter API key (or set OPENROUTER_API_KEY env var)')
    parser.add_argument('--save-template', type=str, help='Save a template config to file and exit')
    
    args = parser.parse_args()
    
    # Save template if requested
    if args.save_template:
        template_config = create_experiment_config(args.template)
        with open(args.save_template, 'w') as f:
            json.dump(template_config, f, indent=2)
        print(f"Template saved to {args.save_template}")
        return
    
    # Get API key
    api_key = args.api_key or os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("Error: OpenRouter API key required. Set OPENROUTER_API_KEY env var or use --api-key")
        return
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
    else:
        config_dict = create_experiment_config(args.template)
    
    config = ExperimentConfig(config_dict)
    
    # Run experiment
    try:
        runner = BatchRunner(config, api_key)
        await runner.run_experiment()
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
    except Exception as e:
        print(f"Experiment failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())