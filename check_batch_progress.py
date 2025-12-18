#!/usr/bin/env python3
"""
Rich Terminal UI Batch Monitor for Secret Hitler LLM Experiments.

Features:
- Color-coded status indicators
- Live-updating games table
- Rich progress bars
- Activity log with recent events
- Policy board visualization

Usage: python check_batch_progress.py [--watch]

Author: Samuel Chakwera (stchakdev)
"""
import os
import sys
from datetime import datetime
from pathlib import Path
import time
import argparse
import json
import re
from collections import deque

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from rich import box


# Simple ASCII Art Banner (works in all terminals)
BANNER = r"""
[bold red]╔═══════════════════════════════════════════════════════════════════════╗
║   ___  ____  ___  ____  ____  ____    _  _  ____  ____  __    ____  ____  ║
║  / __)( ___)/ __)(  _ \( ___)(_  _)  ( )( )(_  _)(_  _)(  )  ( ___)(  _ \ ║
║  \__ \ )__)( (__  )   / )__)   )(     )__(  _)(_   )(   )(__  )__)  )   / ║
║  (___/(____)\___)(__)\_)(__)  (__)   (__)__)(____)(__) (____)(____)(_)\_) ║
╚═══════════════════════════════════════════════════════════════════════════╝[/bold red]
[bold yellow]              LLM DECEPTION DETECTION EXPERIMENT MONITOR[/bold yellow]
[dim]                       Strategic AI Behavior Analysis[/dim]
"""


class GameState:
    """Parsed state of a single game."""

    def __init__(self, game_dir: Path):
        self.game_id = game_dir.name
        self.game_dir = game_dir
        self.log_path = game_dir / "game.log"
        self.phase = "unknown"
        self.turn = 0
        self.liberal_policies = 0
        self.fascist_policies = 0
        self.is_complete = False
        self.winner = None
        self.win_condition = None
        self.president = None
        self.chancellor = None
        self.log_lines = 0
        self.last_modified = None
        self.start_time = None
        self.duration = None
        self.has_error = False
        self.last_action = None

        self._parse()

    def _parse(self):
        """Parse game log to extract state."""
        if not self.log_path.exists():
            return

        try:
            stat = self.log_path.stat()
            self.last_modified = datetime.fromtimestamp(stat.st_mtime)

            with open(self.log_path, 'r') as f:
                content = f.read()
                self.log_lines = content.count('\n')

            # Extract current phase - prefer "new_phase" from events over initial "phase"
            new_phase_matches = list(re.finditer(r'"new_phase":\s*"([^"]+)"', content))
            if new_phase_matches:
                self.phase = new_phase_matches[-1].group(1)
            else:
                # Fallback to action_type for current activity
                action_matches = list(re.finditer(r'"action_type":\s*"([^"]+)"', content))
                if action_matches:
                    self.phase = action_matches[-1].group(1)
                    self.last_action = action_matches[-1].group(1)

            # Extract turn number
            turn_matches = list(re.finditer(r'"turn":\s*(\d+)', content))
            if turn_matches:
                self.turn = int(turn_matches[-1].group(1))

            # Extract policy counts - IMPROVED: check multiple sources
            # 1. First try structured policy_board state
            lib_matches = list(re.finditer(r'"liberal_policies":\s*(\d+)', content))
            fasc_matches = list(re.finditer(r'"fascist_policies":\s*(\d+)', content))
            if lib_matches:
                self.liberal_policies = int(lib_matches[-1].group(1))
            if fasc_matches:
                self.fascist_policies = int(fasc_matches[-1].group(1))

            # 2. If still 0, try to extract from player reasoning text
            # Look for patterns like "4 Fascist policies" or "3 Liberal policies"
            if self.fascist_policies == 0:
                fasc_reasoning = list(re.finditer(r'(\d+)\s+[Ff]ascist\s+polic(?:y|ies)', content))
                if fasc_reasoning:
                    # Get the highest mentioned value (most recent game state)
                    self.fascist_policies = max(int(m.group(1)) for m in fasc_reasoning)

            if self.liberal_policies == 0:
                lib_reasoning = list(re.finditer(r'(\d+)\s+[Ll]iberal\s+polic(?:y|ies)', content))
                if lib_reasoning:
                    self.liberal_policies = max(int(m.group(1)) for m in lib_reasoning)

            # 3. Also check for policy enactment events
            lib_enacted = len(re.findall(r'"policy":\s*"liberal"', content, re.IGNORECASE))
            fasc_enacted = len(re.findall(r'"policy":\s*"fascist"', content, re.IGNORECASE))
            if lib_enacted > self.liberal_policies:
                self.liberal_policies = lib_enacted
            if fasc_enacted > self.fascist_policies:
                self.fascist_policies = fasc_enacted

            # Check completion
            if 'duration_seconds' in content or '"event": "game_end"' in content:
                self.is_complete = True

            # Also check for win conditions in content
            if '"winner":' in content and '"winner": null' not in content:
                winner_check = re.search(r'"winner":\s*"(liberal|fascist)"', content)
                if winner_check:
                    self.is_complete = True

            # Extract winner
            winner_matches = list(re.finditer(r'"winner":\s*"([^"]+)"', content))
            if winner_matches:
                winner = winner_matches[-1].group(1)
                if winner not in ["null", ""]:
                    self.winner = winner

            # Win condition
            cond_matches = list(re.finditer(r'"win_condition":\s*"([^"]+)"', content))
            if cond_matches:
                cond = cond_matches[-1].group(1)
                if cond not in ["null", ""]:
                    self.win_condition = cond

            # Government - look for nomination events
            pres_matches = list(re.finditer(r'"president":\s*"([^"]+)"', content))
            chan_matches = list(re.finditer(r'"chancellor":\s*"([^"]+)"', content))
            if pres_matches:
                pres = pres_matches[-1].group(1)
                if pres not in ["null", ""]:
                    self.president = pres
            if chan_matches:
                chan = chan_matches[-1].group(1)
                if chan not in ["null", ""]:
                    self.chancellor = chan

            # Extract start time from first log entry
            start_matches = list(re.finditer(r'"timestamp":\s*"([^"]+)"', content))
            if start_matches:
                try:
                    self.start_time = datetime.fromisoformat(start_matches[0].group(1).replace('Z', '+00:00'))
                except:
                    pass

            # Calculate duration
            if self.start_time and self.last_modified:
                self.duration = (self.last_modified - self.start_time.replace(tzinfo=None)).total_seconds()

            # Check for REAL errors only (not the word "error" in reasoning)
            real_error_patterns = [
                r'Traceback \(most recent call last\)',
                r'Exception:',
                r'ERROR:',
                r'KeyError:',
                r'ValueError:',
                r'TypeError:',
                r'AttributeError:',
                r'API.*failed',
                r'Connection.*refused',
            ]
            for pattern in real_error_patterns:
                if re.search(pattern, content):
                    self.has_error = True
                    break

        except Exception as e:
            self.has_error = True


class BatchMonitor:
    """Rich terminal UI for monitoring batch experiments."""

    def __init__(self):
        self.console = Console()
        self.logs_dir = Path(__file__).parent / "logs"
        self.batch_metadata = None
        self.start_time = None
        self.target_games = 100
        self.activity_log = deque(maxlen=6)
        self.last_game_count = 0
        self.last_completed_count = 0
        self.logged_games = set()
        self.last_phases = {}  # Track phase changes per game

    def show_banner(self):
        """Display the startup banner."""
        self.console.print(BANNER)
        self.console.print()

    def load_batch_metadata(self):
        """Load batch metadata from .current_batch file."""
        metadata_file = self.logs_dir / ".current_batch"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    self.batch_metadata = json.load(f)
                    self.start_time = datetime.strptime(
                        self.batch_metadata['start_time'],
                        '%Y-%m-%d %H:%M:%S'
                    )
                    self.target_games = self.batch_metadata.get('target_games', 100)
                    return True
            except Exception as e:
                pass
        return False

    def get_games(self) -> list:
        """Get all game states sorted by modification time."""
        if not self.logs_dir.exists():
            return []

        games = []
        for game_dir in self.logs_dir.iterdir():
            if not game_dir.is_dir() or game_dir.name.startswith('.'):
                continue

            game_log = game_dir / "game.log"
            if not game_log.exists():
                continue

            # Filter by start time if available
            if self.start_time:
                mtime = datetime.fromtimestamp(game_log.stat().st_mtime)
                if mtime < self.start_time:
                    continue

            games.append(GameState(game_dir))

        # Sort by modification time (newest first)
        games.sort(key=lambda g: g.last_modified or datetime.min, reverse=True)
        return games

    def format_duration(self, seconds: float) -> str:
        """Format duration as MM:SS or HH:MM:SS."""
        if seconds is None:
            return "--:--"
        minutes, secs = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def create_header(self, games: list) -> Panel:
        """Create header panel with batch info."""
        completed = sum(1 for g in games if g.is_complete)
        elapsed = datetime.now() - self.start_time if self.start_time else None

        # Calculate ETA
        eta_str = "Calculating..."
        rate_str = "0.0"
        if completed > 0 and elapsed:
            elapsed_hours = elapsed.total_seconds() / 3600
            rate = completed / elapsed_hours if elapsed_hours > 0 else 0
            rate_str = f"{rate:.1f}"
            if rate > 0:
                remaining = self.target_games - completed
                eta_time = datetime.now() + (elapsed * (remaining / completed)) if completed > 0 else None
                if eta_time:
                    eta_str = eta_time.strftime('%H:%M:%S')

        batch_tag = self.batch_metadata.get('batch_tag', 'Unknown') if self.batch_metadata else 'Unknown'
        model = self.batch_metadata.get('model', 'Unknown') if self.batch_metadata else 'Unknown'

        # Truncate model name
        if len(model) > 30:
            model = model[:27] + "..."

        header_text = Text()
        header_text.append("Batch: ", style="dim")
        header_text.append(batch_tag, style="cyan bold")
        header_text.append(" | Model: ", style="dim")
        header_text.append(model, style="cyan")
        header_text.append("\n")

        if elapsed:
            elapsed_str = self.format_duration(elapsed.total_seconds())
            header_text.append("Started: ", style="dim")
            header_text.append(self.start_time.strftime('%H:%M:%S'), style="white")
            header_text.append(" | Elapsed: ", style="dim")
            header_text.append(elapsed_str, style="white")
            header_text.append(" | Rate: ", style="dim")
            header_text.append(f"{rate_str}/hr", style="green")
            header_text.append(" | ETA: ", style="dim")
            header_text.append(eta_str, style="yellow")

        return Panel(
            header_text,
            title="[bold blue]SECRET HITLER BATCH MONITOR[/bold blue]",
            border_style="blue",
            box=box.DOUBLE
        )

    def create_progress_bar(self, games: list) -> Text:
        """Create a text-based progress bar."""
        completed = sum(1 for g in games if g.is_complete)
        progress = completed / self.target_games if self.target_games > 0 else 0

        bar_width = 40
        filled = int(bar_width * progress)
        bar = "━" * filled + "╺" + "─" * max(0, bar_width - filled - 1)

        text = Text()
        text.append("Progress ", style="dim")
        text.append(bar, style="green" if progress > 0.5 else "yellow")
        text.append(f" {progress*100:.0f}% ", style="bold")
        text.append(f"{completed}/{self.target_games} games", style="dim")

        return text

    def create_games_table(self, games: list) -> Table:
        """Create table showing recent games."""
        table = Table(
            title="Recent Games",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
            border_style="dim",
            expand=True
        )

        table.add_column("#", style="dim", width=3, justify="right")
        table.add_column("Status", width=12)
        table.add_column("Phase/Result", width=16)
        table.add_column("Turn", width=5, justify="center")
        table.add_column("Time", width=8, justify="center")
        table.add_column("Policies", width=18)

        # Show last 5 games
        for i, game in enumerate(games[:5], 1):
            # Status - fixed error detection
            if game.is_complete:
                status = Text("Complete", style="green bold")
            elif game.has_error:
                status = Text("Error", style="red bold")
            else:
                status = Text("Running", style="yellow")

            # Result/Phase
            if game.is_complete and game.winner:
                if 'liberal' in game.winner.lower():
                    result = Text("Liberal Win", style="blue bold")
                else:
                    result = Text("Fascist Win", style="red bold")
            elif not game.is_complete:
                phase = game.phase.replace('_', ' ').title()
                if len(phase) > 14:
                    phase = phase[:13] + "..."
                result = Text(phase, style="white")
            else:
                result = Text("Unknown", style="dim")

            # Turn
            turn = Text(str(game.turn) if game.turn > 0 else "-", style="white")

            # Time
            duration = self.format_duration(game.duration)

            # Policies - visual representation with numbers
            lib = game.liberal_policies
            fasc = game.fascist_policies

            lib_filled = "●" * min(lib, 5)
            lib_empty = "○" * max(0, 5 - lib)
            fasc_filled = "●" * min(fasc, 6)
            fasc_empty = "○" * max(0, 6 - fasc)

            policies = Text()
            policies.append(lib_filled, style="blue bold")
            policies.append(lib_empty, style="blue dim")
            policies.append(" ", style="dim")
            policies.append(fasc_filled, style="red bold")
            policies.append(fasc_empty, style="red dim")

            table.add_row(
                str(i),
                status,
                result,
                turn,
                duration,
                policies
            )

            # Add government info for running games
            if not game.is_complete and (game.president or game.chancellor):
                gov_text = Text()
                gov_text.append("  Gov: ", style="dim")
                if game.president:
                    gov_text.append(game.president[:10], style="cyan")
                gov_text.append(" → ", style="dim")
                if game.chancellor:
                    gov_text.append(game.chancellor[:10], style="cyan")
                table.add_row("", "", gov_text, "", "", "")

        return table

    def create_activity_log(self) -> Panel:
        """Create activity log panel."""
        if not self.activity_log:
            log_text = Text("Waiting for game activity...", style="dim italic")
        else:
            log_text = Text()
            for entry in list(self.activity_log):
                log_text.append(entry + "\n")

        return Panel(
            log_text,
            title="[bold]Activity Log[/bold]",
            border_style="dim",
            box=box.ROUNDED,
            height=8
        )

    def update_activity(self, games: list):
        """Update activity log with new events."""
        time_str = datetime.now().strftime('%H:%M:%S')

        # Track new games starting
        if len(games) > self.last_game_count:
            new_count = len(games) - self.last_game_count
            self.activity_log.append(
                f"[dim]{time_str}[/dim] [green]Game {len(games)} started[/green]"
            )

        # Log completed games
        completed_count = sum(1 for g in games if g.is_complete)
        if completed_count > self.last_completed_count:
            for game in games:
                if game.is_complete and game.game_id not in self.logged_games:
                    winner = game.winner or "Unknown"
                    style = "blue" if 'liberal' in winner.lower() else "red"
                    duration = self.format_duration(game.duration)

                    # Add win condition if available
                    win_info = ""
                    if game.win_condition:
                        win_info = f" ({game.win_condition.replace('_', ' ')})"

                    self.activity_log.append(
                        f"[dim]{time_str}[/dim] [{style}]{winner.title()} win{win_info}[/{style}] {duration}"
                    )
                    self.logged_games.add(game.game_id)

        # Log phase changes for running games
        for game in games:
            if not game.is_complete:
                prev_phase = self.last_phases.get(game.game_id)
                if prev_phase and prev_phase != game.phase:
                    phase_name = game.phase.replace('_', ' ').title()
                    if 'legislative' in game.phase.lower():
                        self.activity_log.append(
                            f"[dim]{time_str}[/dim] [yellow]Legislative session[/yellow] ({game.president} → {game.chancellor})"
                        )
                    elif 'nomination' in game.phase.lower():
                        self.activity_log.append(
                            f"[dim]{time_str}[/dim] [cyan]New nomination[/cyan]"
                        )
                    elif 'vote' in game.phase.lower():
                        self.activity_log.append(
                            f"[dim]{time_str}[/dim] [white]Voting...[/white]"
                        )
                self.last_phases[game.game_id] = game.phase

        self.last_completed_count = completed_count
        self.last_game_count = len(games)

    def create_layout(self, games: list) -> Layout:
        """Create the full dashboard layout."""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=6),
            Layout(name="progress", size=3),
            Layout(name="games", size=14),
            Layout(name="activity", size=8),
            Layout(name="footer", size=1)
        )

        layout["header"].update(self.create_header(games))
        layout["progress"].update(Panel(self.create_progress_bar(games), box=box.SIMPLE))
        layout["games"].update(self.create_games_table(games))
        layout["activity"].update(self.create_activity_log())

        footer_text = Text()
        footer_text.append("Press ", style="dim")
        footer_text.append("Ctrl+C", style="bold yellow")
        footer_text.append(" to exit | Last update: ", style="dim")
        footer_text.append(datetime.now().strftime('%H:%M:%S'), style="white")
        layout["footer"].update(footer_text)

        return layout

    def show_completion_banner(self, games: list):
        """Show a big completion banner with summary."""
        completed = sum(1 for g in games if g.is_complete)
        liberal_wins = sum(1 for g in games if g.is_complete and g.winner and 'liberal' in g.winner.lower())
        fascist_wins = completed - liberal_wins

        elapsed = datetime.now() - self.start_time if self.start_time else None
        elapsed_str = self.format_duration(elapsed.total_seconds()) if elapsed else "Unknown"

        # Ring terminal bell
        print('\a')  # Terminal bell

        banner = f"""
[bold green]
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║   ██████╗  █████╗ ████████╗ ██████╗██╗  ██╗                          ║
║   ██╔══██╗██╔══██╗╚══██╔══╝██╔════╝██║  ██║                          ║
║   ██████╔╝███████║   ██║   ██║     ███████║                          ║
║   ██╔══██╗██╔══██║   ██║   ██║     ██╔══██║                          ║
║   ██████╔╝██║  ██║   ██║   ╚██████╗██║  ██║                          ║
║   ╚═════╝ ╚═╝  ╚═╝   ╚═╝    ╚═════╝╚═╝  ╚═╝                          ║
║                                                                       ║
║    ██████╗ ██████╗ ███╗   ███╗██████╗ ██╗     ███████╗████████╗███████╗║
║   ██╔════╝██╔═══██╗████╗ ████║██╔══██╗██║     ██╔════╝╚══██╔══╝██╔════╝║
║   ██║     ██║   ██║██╔████╔██║██████╔╝██║     █████╗     ██║   █████╗  ║
║   ██║     ██║   ██║██║╚██╔╝██║██╔═══╝ ██║     ██╔══╝     ██║   ██╔══╝  ║
║   ╚██████╗╚██████╔╝██║ ╚═╝ ██║██║     ███████╗███████╗   ██║   ███████╗║
║    ╚═════╝ ╚═════╝ ╚═╝     ╚═╝╚═╝     ╚══════╝╚══════╝   ╚═╝   ╚══════╝║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
[/bold green]

[bold white]                    🎮 BATCH EXPERIMENT FINISHED 🎮[/bold white]

[bold cyan]═══════════════════════════════════════════════════════════════════════[/bold cyan]

  [bold]Total Games:[/bold]     {completed}
  [bold]Duration:[/bold]        {elapsed_str}

  [blue bold]Liberal Wins:[/blue bold]    {liberal_wins} ({liberal_wins/completed*100:.0f}%)
  [red bold]Fascist Wins:[/red bold]    {fascist_wins} ({fascist_wins/completed*100:.0f}%)

[bold cyan]═══════════════════════════════════════════════════════════════════════[/bold cyan]

[dim]Results saved to database. Run analysis with:[/dim]
[yellow]  python -m analytics.statistical_analysis[/yellow]

"""
        self.console.print(banner)

    def run(self, watch: bool = True, interval: int = 3):
        """Run the batch monitor."""
        # Show banner
        self.show_banner()

        if not self.load_batch_metadata():
            self.console.print("[red]No batch metadata found![/red]")
            self.console.print("[dim]Run a batch first with:[/dim] python run_game.py --batch --games N")
            return

        self.console.print(f"[green]Loaded batch:[/green] {self.batch_metadata.get('batch_tag', 'Unknown')}")
        self.console.print(f"[dim]Target: {self.target_games} games | Refresh: {interval}s[/dim]\n")
        time.sleep(1)

        batch_completed = False

        if watch:
            try:
                with Live(console=self.console, refresh_per_second=1, screen=True) as live:
                    while True:
                        games = self.get_games()
                        self.update_activity(games)
                        live.update(self.create_layout(games))

                        # Check for batch completion
                        completed = sum(1 for g in games if g.is_complete)
                        if completed >= self.target_games and not batch_completed:
                            batch_completed = True
                            # Add completion to activity log
                            time_str = datetime.now().strftime('%H:%M:%S')
                            self.activity_log.append(
                                f"[dim]{time_str}[/dim] [bold green]🎉 BATCH COMPLETE! {completed}/{self.target_games} games[/bold green]"
                            )
                            # Ring bell multiple times
                            for _ in range(3):
                                print('\a', end='', flush=True)
                                time.sleep(0.3)

                        time.sleep(interval)
            except KeyboardInterrupt:
                pass

            # Show completion summary if batch finished
            games = self.get_games()
            completed = sum(1 for g in games if g.is_complete)
            if completed >= self.target_games:
                self.show_completion_banner(games)
            else:
                self.console.print("\n[yellow]Monitoring stopped.[/yellow]")
        else:
            games = self.get_games()
            completed = sum(1 for g in games if g.is_complete)
            if completed >= self.target_games:
                self.show_completion_banner(games)
            else:
                self.console.print(self.create_layout(games))


def main():
    parser = argparse.ArgumentParser(
        description='Secret Hitler LLM Experiment Monitor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python check_batch_progress.py --watch        # Live monitoring
  python check_batch_progress.py                # Single snapshot
  python check_batch_progress.py --interval 5   # Slower refresh (5 seconds)
        """
    )
    parser.add_argument('--watch', '-w', action='store_true',
                        help='Watch progress in real-time')
    parser.add_argument('--interval', '-i', type=int, default=3,
                        help='Update interval in seconds (default: 3)')

    args = parser.parse_args()

    monitor = BatchMonitor()
    monitor.run(watch=args.watch, interval=args.interval)


if __name__ == '__main__':
    main()
