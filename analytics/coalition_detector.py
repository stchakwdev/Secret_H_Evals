"""
Coalition detection for Secret Hitler game analysis.

Uses community detection algorithms to identify voting coalitions
and measures how well detected coalitions align with true team affiliations.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import community as community_louvain


@dataclass
class CoalitionResult:
    """Container for coalition detection results."""
    partition: Dict[str, int]  # player_name -> coalition_id
    num_coalitions: int
    purity_score: float
    modularity: float
    coalition_sizes: Dict[int, int]
    coalition_compositions: Dict[int, Dict[str, int]]  # coalition_id -> {role: count}


class CoalitionDetector:
    """
    Detect voting coalitions using community detection algorithms.

    Uses voting alignment patterns to build a network where players who
    vote together are connected, then applies Louvain community detection
    to identify coalitions.
    """

    def __init__(self, min_alignment_threshold: float = 0.5):
        """
        Initialize coalition detector.

        Args:
            min_alignment_threshold: Minimum vote alignment to create edge (0-1)
        """
        self.min_alignment_threshold = min_alignment_threshold

    def build_vote_alignment_matrix(
        self,
        votes: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Build vote alignment matrix from voting records.

        Args:
            votes: List of voting round dictionaries with format:
                {
                    'round': int,
                    'votes': {player_name: bool, ...}  # True = Ja, False = Nein
                }

        Returns:
            Tuple of (alignment_matrix, player_names)
            alignment_matrix[i,j] = proportion of times i and j voted the same
        """
        if not votes:
            return np.array([]), []

        # Get all unique players
        all_players = set()
        for vote_round in votes:
            all_players.update(vote_round.get('votes', {}).keys())

        player_names = sorted(all_players)
        n_players = len(player_names)
        player_idx = {name: i for i, name in enumerate(player_names)}

        # Count agreements
        agreement_counts = np.zeros((n_players, n_players))
        total_rounds = np.zeros((n_players, n_players))

        for vote_round in votes:
            round_votes = vote_round.get('votes', {})
            players_in_round = list(round_votes.keys())

            for i, p1 in enumerate(players_in_round):
                for p2 in players_in_round[i + 1:]:
                    idx1, idx2 = player_idx[p1], player_idx[p2]

                    # Count this round
                    total_rounds[idx1, idx2] += 1
                    total_rounds[idx2, idx1] += 1

                    # Check agreement
                    if round_votes[p1] == round_votes[p2]:
                        agreement_counts[idx1, idx2] += 1
                        agreement_counts[idx2, idx1] += 1

        # Calculate alignment proportions
        with np.errstate(divide='ignore', invalid='ignore'):
            alignment_matrix = np.where(
                total_rounds > 0,
                agreement_counts / total_rounds,
                0.0
            )

        # Set diagonal to 1 (perfect self-agreement)
        np.fill_diagonal(alignment_matrix, 1.0)

        return alignment_matrix, player_names

    def detect_coalitions(
        self,
        alignment_matrix: np.ndarray,
        player_names: List[str],
        resolution: float = 1.0
    ) -> Dict[str, int]:
        """
        Detect coalitions using Louvain community detection.

        Args:
            alignment_matrix: Vote alignment matrix from build_vote_alignment_matrix
            player_names: List of player names corresponding to matrix indices
            resolution: Louvain resolution parameter (higher = more communities)

        Returns:
            Dictionary mapping player_name -> coalition_id
        """
        if len(player_names) == 0:
            return {}

        # Build network graph
        G = nx.Graph()
        G.add_nodes_from(player_names)

        n = len(player_names)
        for i in range(n):
            for j in range(i + 1, n):
                weight = alignment_matrix[i, j]
                if weight >= self.min_alignment_threshold:
                    G.add_edge(
                        player_names[i],
                        player_names[j],
                        weight=weight
                    )

        # Run Louvain algorithm
        if G.number_of_edges() == 0:
            # No edges - each player in own coalition
            return {name: i for i, name in enumerate(player_names)}

        partition = community_louvain.best_partition(
            G,
            weight='weight',
            resolution=resolution,
            random_state=42
        )

        return partition

    def calculate_coalition_purity(
        self,
        partition: Dict[str, int],
        true_roles: Dict[str, str]
    ) -> float:
        """
        Calculate how well coalitions match true team affiliations.

        Purity score measures whether players in the same coalition
        belong to the same team (liberal vs fascist).

        Args:
            partition: Coalition assignment from detect_coalitions
            true_roles: Actual roles {player_name: 'liberal'/'fascist'/'hitler'}

        Returns:
            Purity score from 0 to 1 (1 = perfect team separation)
        """
        if not partition:
            return 0.0

        # Normalize roles (Hitler counts as fascist team)
        team_mapping = {
            'liberal': 'liberal',
            'fascist': 'fascist',
            'hitler': 'fascist'
        }

        # Group players by coalition
        coalitions: Dict[int, List[str]] = {}
        for player, coalition_id in partition.items():
            if coalition_id not in coalitions:
                coalitions[coalition_id] = []
            coalitions[coalition_id].append(player)

        # Calculate purity for each coalition
        purity_scores = []
        weights = []

        for coalition_id, members in coalitions.items():
            if not members:
                continue

            # Count team memberships
            team_counts = {'liberal': 0, 'fascist': 0}
            for member in members:
                role = true_roles.get(member, 'unknown')
                team = team_mapping.get(role)
                if team:
                    team_counts[team] += 1

            total_known = team_counts['liberal'] + team_counts['fascist']
            if total_known > 0:
                # Purity = proportion of dominant team
                dominant_count = max(team_counts.values())
                purity = dominant_count / total_known
                purity_scores.append(purity)
                weights.append(total_known)

        if not purity_scores:
            return 0.0

        # Weighted average by coalition size
        weighted_purity = np.average(purity_scores, weights=weights)
        return float(weighted_purity)

    def calculate_modularity(
        self,
        alignment_matrix: np.ndarray,
        player_names: List[str],
        partition: Dict[str, int]
    ) -> float:
        """
        Calculate modularity score for the detected partition.

        Modularity measures the quality of the community structure.
        Higher values indicate stronger community separation.

        Returns:
            Modularity score (typically 0 to 1)
        """
        if len(player_names) == 0:
            return 0.0

        # Build graph
        G = nx.Graph()
        G.add_nodes_from(player_names)

        n = len(player_names)
        for i in range(n):
            for j in range(i + 1, n):
                weight = alignment_matrix[i, j]
                if weight >= self.min_alignment_threshold:
                    G.add_edge(player_names[i], player_names[j], weight=weight)

        if G.number_of_edges() == 0:
            return 0.0

        return community_louvain.modularity(partition, G, weight='weight')

    def analyze_game_coalitions(
        self,
        votes: List[Dict[str, Any]],
        true_roles: Dict[str, str],
        resolution: float = 1.0
    ) -> CoalitionResult:
        """
        Full coalition analysis for a single game.

        Args:
            votes: List of voting round data
            true_roles: Actual player roles
            resolution: Louvain resolution parameter

        Returns:
            CoalitionResult with all metrics
        """
        # Build alignment matrix
        alignment_matrix, player_names = self.build_vote_alignment_matrix(votes)

        if len(player_names) == 0:
            return CoalitionResult(
                partition={},
                num_coalitions=0,
                purity_score=0.0,
                modularity=0.0,
                coalition_sizes={},
                coalition_compositions={}
            )

        # Detect coalitions
        partition = self.detect_coalitions(
            alignment_matrix, player_names, resolution
        )

        # Calculate metrics
        purity = self.calculate_coalition_purity(partition, true_roles)
        modularity = self.calculate_modularity(
            alignment_matrix, player_names, partition
        )

        # Coalition sizes
        coalition_sizes: Dict[int, int] = {}
        for coalition_id in partition.values():
            coalition_sizes[coalition_id] = coalition_sizes.get(coalition_id, 0) + 1

        # Coalition compositions
        team_mapping = {'liberal': 'liberal', 'fascist': 'fascist', 'hitler': 'fascist'}
        coalition_compositions: Dict[int, Dict[str, int]] = {}

        for player, coalition_id in partition.items():
            if coalition_id not in coalition_compositions:
                coalition_compositions[coalition_id] = {'liberal': 0, 'fascist': 0}

            role = true_roles.get(player, 'unknown')
            team = team_mapping.get(role)
            if team:
                coalition_compositions[coalition_id][team] += 1

        return CoalitionResult(
            partition=partition,
            num_coalitions=len(coalition_sizes),
            purity_score=purity,
            modularity=modularity,
            coalition_sizes=coalition_sizes,
            coalition_compositions=coalition_compositions
        )

    def analyze_batch_coalitions(
        self,
        games_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze coalitions across multiple games.

        Args:
            games_data: List of game dictionaries with keys:
                - votes: voting history
                - roles: player role assignments

        Returns:
            Aggregated statistics across all games
        """
        results = []

        for game in games_data:
            votes = game.get('votes', [])
            roles = game.get('roles', {})

            if votes and roles:
                result = self.analyze_game_coalitions(votes, roles)
                results.append(result)

        if not results:
            return {"error": "No valid games to analyze"}

        # Aggregate statistics
        purity_scores = [r.purity_score for r in results]
        modularity_scores = [r.modularity for r in results]
        num_coalitions = [r.num_coalitions for r in results]

        return {
            "n_games": len(results),
            "purity": {
                "mean": np.mean(purity_scores),
                "std": np.std(purity_scores),
                "min": np.min(purity_scores),
                "max": np.max(purity_scores)
            },
            "modularity": {
                "mean": np.mean(modularity_scores),
                "std": np.std(modularity_scores)
            },
            "num_coalitions": {
                "mean": np.mean(num_coalitions),
                "mode": int(np.argmax(np.bincount(num_coalitions)))
            }
        }


def get_alignment_network_for_visualization(
    alignment_matrix: np.ndarray,
    player_names: List[str],
    partition: Dict[str, int],
    true_roles: Optional[Dict[str, str]] = None,
    min_edge_weight: float = 0.3
) -> Dict[str, Any]:
    """
    Prepare network data for visualization (Plotly/D3).

    Args:
        alignment_matrix: Vote alignment matrix
        player_names: Player names
        partition: Coalition assignments
        true_roles: Optional actual roles for coloring
        min_edge_weight: Minimum edge weight to include

    Returns:
        Dictionary with nodes and edges for visualization
    """
    nodes = []
    edges = []

    # Role to color mapping
    role_colors = {
        'liberal': '#3498db',    # Blue
        'fascist': '#e74c3c',    # Red
        'hitler': '#2c3e50',     # Dark
        'unknown': '#95a5a6'     # Gray
    }

    # Create nodes
    for i, name in enumerate(player_names):
        role = true_roles.get(name, 'unknown') if true_roles else 'unknown'
        coalition = partition.get(name, 0)

        nodes.append({
            'id': name,
            'label': name,
            'coalition': coalition,
            'role': role,
            'color': role_colors.get(role, '#95a5a6'),
            'x': np.cos(2 * np.pi * i / len(player_names)),  # Circle layout
            'y': np.sin(2 * np.pi * i / len(player_names))
        })

    # Create edges
    n = len(player_names)
    for i in range(n):
        for j in range(i + 1, n):
            weight = alignment_matrix[i, j]
            if weight >= min_edge_weight:
                # Edge color based on whether same coalition
                same_coalition = partition.get(player_names[i]) == partition.get(player_names[j])

                edges.append({
                    'source': player_names[i],
                    'target': player_names[j],
                    'weight': float(weight),
                    'same_coalition': same_coalition,
                    'color': '#2ecc71' if same_coalition else '#bdc3c7'
                })

    return {
        'nodes': nodes,
        'edges': edges,
        'num_coalitions': len(set(partition.values()))
    }
