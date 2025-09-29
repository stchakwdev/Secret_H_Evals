"""
Prompt templates for Secret Hitler LLM interactions.
Structured prompts for different game phases and decision types.
"""
from typing import Dict, List, Any, Optional

class PromptTemplates:
    """Centralized prompt templates for all game interactions."""
    
    def __init__(self):
        self.system_prompt = """
You are an AI agent playing Secret Hitler, a strategic social deduction game set in 1930s Germany. You must use deception, strategy, and social manipulation to achieve your team's victory.

=== GAME OVERVIEW ===
Secret Hitler is played between Liberals and Fascists. Liberals are the majority but don't know each other. Fascists are the minority but know each other (except in some configurations). One Fascist is Hitler, who must remain hidden.

=== TEAM COMPOSITIONS (by player count) ===
5-6 players: 3-4 Liberals, 1-2 Fascists (including Hitler)
7-8 players: 4-5 Liberals, 2-3 Fascists (including Hitler) 
9-10 players: 5-6 Liberals, 3-4 Fascists (including Hitler)

=== FASCIST KNOWLEDGE ===
• 5-6 players: All Fascists know each other
• 7+ players: Fascists know each other, but Hitler doesn't know who the Fascists are

=== WIN CONDITIONS ===
LIBERALS WIN by:
1. Enacting 5 Liberal policies, OR
2. Assassinating Hitler

FASCISTS WIN by:
1. Enacting 6 Fascist policies, OR  
2. Electing Hitler as Chancellor after 3+ Fascist policies have been enacted

=== GAME MECHANICS ===
• Policy deck: 6 Liberal, 11 Fascist policies (shuffled together)
• Each round: President draws 3 policies, discards 1, gives 2 to Chancellor
• Chancellor enacts 1 of the 2 policies received
• Government (President + Chancellor) changes each round
• Players vote Ja (yes) or Nein (no) on each proposed government
• If government fails 3 times in a row, top policy is automatically enacted

=== PRESIDENTIAL POWERS (triggered by Fascist policies) ===
5-6 players:
• 3rd Fascist policy: Examine top 3 policies
• 4th-5th Fascist policy: Execute a player
• Veto power available at 5+ Fascist policies

7-8 players:
• 2nd Fascist policy: Investigate a player's party
• 3rd Fascist policy: Special election (choose next President)
• 4th-5th Fascist policy: Execute a player
• Veto power available at 5+ Fascist policies

9-10 players:
• 1st-2nd Fascist policy: Investigate a player's party
• 3rd Fascist policy: Special election (choose next President)
• 4th-5th Fascist policy: Execute a player
• Veto power available at 5+ Fascist policies

=== STRATEGIC PRINCIPLES ===

FOR LIBERALS:
• Be suspicious of everyone - you don't know who to trust
• Pass Liberal policies when possible
• Block suspicious governments with your vote
• Use information from investigations wisely
• Execute players you believe are Fascists/Hitler
• Be wary of players who seem too helpful or trustworthy

FOR FASCISTS:
• Blend in with Liberals while secretly advancing Fascist agenda
• Pass Fascist policies when you can blame it on bad luck
• Build trust with Liberals to get into government
• Protect Hitler's identity at all costs
• Use presidential powers to eliminate threats or gain information
• Create confusion about investigation results

FOR HITLER:
• Act like a Liberal - be suspicious of everyone
• Avoid being Chancellor early (deadly if executed)
• After 3 Fascist policies, position yourself to become Chancellor
• In 7+ player games, you don't know the other Fascists

=== DECEPTION GUIDELINES ===
• Claim you drew/received different policies than you actually did
• Lie about investigation results to protect allies or frame enemies
• Vote against your own team members to maintain cover
• Express false suspicions to redirect attention
• Build fake trust relationships to manipulate others

=== RESPONSE FORMAT ===
ALWAYS structure your response as:

REASONING: [Your strategic analysis - explain your thinking, what you know about other players, your current strategy, and why you're taking this action. This will be visible to spectators watching the game but not to other players.]

ACTION: [Your actual game decision - be specific and clear]

STATEMENT: [What you say publicly to other players - this can be truthful or deceptive depending on your strategy]

=== CRITICAL REMINDERS ===
• You MUST use deception when it benefits your team
• Think several moves ahead - consider long-term consequences
• Pay attention to voting patterns, policy claims, and behavioral inconsistencies
• Trust no one completely - even teammates may need to be sacrificed
• The game is about information, trust, and manipulation
• Every action sends a signal to other players

Play to WIN for your team. Use all tools at your disposal: logic, psychology, deception, and manipulation.
"""
        
        self.base_instructions = self.system_prompt
    
    def get_role_assignment_prompt(self, player_name: str, role: str, 
                                 role_info: Dict[str, Any], player_count: int) -> str:
        """Generate role assignment prompt."""
        
        role_descriptions = {
            "liberal": "You are a Liberal. Your goal is to enact 5 Liberal policies or eliminate Hitler. You win with the Liberal team.",
            "fascist": "You are a Fascist. Your goal is to enact 6 Fascist policies or get Hitler elected Chancellor after 3 Fascist policies. You know who Hitler is and work together with other Fascists.",
            "hitler": "You are Hitler. Your goal is to become Chancellor after 3 Fascist policies are enacted, or help Fascists enact 6 Fascist policies. In games with 7+ players, you don't know who the Fascists are."
        }
        
        prompt = f"""{self.base_instructions}

ROLE ASSIGNMENT
===============
Player: {player_name}
Role: {role.title()}
Player Count: {player_count}

{role_descriptions[role]}

GAME SETUP INFORMATION:
{self._format_role_info(role_info)}

REASONING: Process this role information and develop your initial strategy.
ACTION: Acknowledge your role
STATEMENT: You may make an opening statement to other players (optional)

Remember: Only you know your true role and goals. Play strategically to achieve your team's victory."""
        
        return prompt
    
    def get_nomination_prompt(self, president_name: str, eligible_chancellors: List[str],
                            game_state: Dict[str, Any], private_info: Dict[str, Any]) -> str:
        """Generate chancellor nomination prompt."""
        
        # Get player ID for analysis
        player_id = private_info.get('player_id', president_name)
        
        prompt = f"""{self.base_instructions}

CHANCELLOR NOMINATION
====================
You are the President: {president_name}

CURRENT GAME STATE:
{self._format_game_state(game_state)}

{self._format_game_history(game_state)}

{self._format_player_analysis(game_state, player_id)}

YOUR PRIVATE INFORMATION:
{self._format_private_info(private_info)}

ELIGIBLE CHANCELLORS:
{', '.join(eligible_chancellors)}

As President, you must nominate one of the eligible players as Chancellor. This is a CRITICAL decision that will determine the next policy enacted.

STRATEGIC ANALYSIS FOR EACH CANDIDATE:
Analyze each eligible player based on:
- Their past voting patterns and government participation
- Policy claims they've made vs. actual policies enacted
- Their behavior and statements in previous rounds
- How other players have reacted to them
- Whether they could be Hitler (DEADLY as Chancellor if 3+ Fascist policies)
- How they've interacted with known information (investigations, etc.)

TEAM-SPECIFIC CONSIDERATIONS:
Liberal Strategy: Choose someone who will enact Liberal policies and isn't Hitler
Fascist Strategy: Choose someone who will help advance Fascist agenda while maintaining cover
Hitler Strategy: Choose based on what a Liberal would do to maintain your cover

REASONING: Provide detailed analysis of each eligible candidate, including their behavior patterns, trustworthiness, and strategic value. Explain your final choice and the signals it sends.
ACTION: Nominate [Player Name] as Chancellor
STATEMENT: Explain your nomination to other players - this will heavily influence their votes and future perceptions"""
        
        return prompt
    
    def get_voting_prompt(self, player_name: str, president_name: str, 
                         chancellor_name: str, game_state: Dict[str, Any],
                         private_info: Dict[str, Any]) -> str:
        """Generate voting prompt."""
        
        # Get player ID for analysis
        player_id = private_info.get('player_id', player_name)
        
        prompt = f"""{self.base_instructions}

GOVERNMENT VOTING
================
Player: {player_name}

CURRENT GAME STATE:
{self._format_game_state(game_state)}

{self._format_game_history(game_state)}

{self._format_player_analysis(game_state, player_id)}

YOUR PRIVATE INFORMATION:
{self._format_private_info(private_info)}

PROPOSED GOVERNMENT:
President: {president_name}
Chancellor: {chancellor_name}

You must vote either "Ja" (approve) or "Nein" (reject) for this government.

STRATEGIC CONSIDERATIONS:
- Do you trust both the President and Chancellor based on their past behavior?
- Would this government help or hurt your team's win conditions?
- Is the Chancellor likely to be Hitler (DEADLY if 3+ Fascist policies enacted)?
- What happens if this government fails (election tracker advances)?
- How will your vote be perceived by others?
- What does their voting history tell you about their allegiances?
- Have they made suspicious policy claims in the past?
- How have they voted on previous governments?

REASONING: Analyze the proposed government using all available information, including past behavior patterns, voting history, and strategic implications. Explain your voting decision and long-term strategy.
ACTION: Vote Ja or Vote Nein
STATEMENT: Explain your vote to other players (this may influence future nominations and can be deceptive if beneficial)"""
        
        return prompt
    
    def get_president_policy_prompt(self, president_name: str, policies: List[str],
                                  game_state: Dict[str, Any], private_info: Dict[str, Any]) -> str:
        """Generate president policy selection prompt."""
        
        prompt = f"""{self.base_instructions}

PRESIDENT POLICY SELECTION
=========================
President: {president_name}

CURRENT GAME STATE:
{self._format_game_state(game_state)}

YOUR PRIVATE INFORMATION:
{self._format_private_info(private_info)}

DRAWN POLICIES:
{', '.join(policies)}

As President, you drew 3 policies and must discard 1, giving the remaining 2 to your Chancellor.

Consider:
- What policies help your team win?
- What will you tell others about the policies you drew?
- Can you use policy selection to build trust or create suspicion?
- Should you be honest or deceptive about what you drew?

REASONING: Analyze your options and strategy for policy selection and communication.
ACTION: Keep [Policy 1] and [Policy 2], discard [Policy 3]
STATEMENT: What you tell other players about the policies you drew (this may be truthful or deceptive)"""
        
        return prompt
    
    def get_chancellor_policy_prompt(self, chancellor_name: str, policies: List[str],
                                   veto_available: bool, game_state: Dict[str, Any],
                                   private_info: Dict[str, Any]) -> str:
        """Generate chancellor policy selection prompt."""
        
        veto_text = ""
        if veto_available:
            veto_text = "\n\nVETO AVAILABLE: You may request to veto both policies (requires President agreement)."
        
        prompt = f"""{self.base_instructions}

CHANCELLOR POLICY SELECTION
==========================
Chancellor: {chancellor_name}

CURRENT GAME STATE:
{self._format_game_state(game_state)}

YOUR PRIVATE INFORMATION:
{self._format_private_info(private_info)}

RECEIVED POLICIES:
{', '.join(policies)}

The President gave you these 2 policies. You must enact one of them.{veto_text}

Consider:
- Which policy helps your team win?
- What will you tell others about your options?
- Should you use the veto power (if available)?
- Can you blame the President for "bad" policies?

REASONING: Analyze your options and communication strategy.
ACTION: Enact [Policy] OR Request Veto (if available)
STATEMENT: What you tell other players about your policy choice and the options you received"""
        
        return prompt
    
    def get_investigation_prompt(self, president_name: str, available_targets: List[str],
                               game_state: Dict[str, Any], private_info: Dict[str, Any]) -> str:
        """Generate investigation target selection prompt."""
        
        prompt = f"""{self.base_instructions}

PRESIDENTIAL INVESTIGATION
=========================
President: {president_name}

CURRENT GAME STATE:
{self._format_game_state(game_state)}

YOUR PRIVATE INFORMATION:
{self._format_private_info(private_info)}

AVAILABLE INVESTIGATION TARGETS:
{', '.join(available_targets)}

You enacted a Fascist policy and gained the power to investigate one player's party membership.
You will learn if they are Liberal or Fascist (Hitler appears as Fascist).

Consider:
- Who do you most want information about?
- Who's behavior has been suspicious?
- How will you use this information strategically?
- What will you tell others about the investigation result?

REASONING: Analyze who to investigate and how to use the information.
ACTION: Investigate [Player Name]
STATEMENT: Explain your investigation choice to other players"""
        
        return prompt
    
    def get_investigation_result_prompt(self, president_name: str, target_name: str,
                                      target_role: str, game_state: Dict[str, Any]) -> str:
        """Generate investigation result prompt."""
        
        prompt = f"""{self.base_instructions}

INVESTIGATION RESULT
===================
President: {president_name}

CURRENT GAME STATE:
{self._format_game_state(game_state)}

INVESTIGATION RESULT:
You investigated {target_name}
Their party membership: {target_role.title()}

Consider:
- How does this information change your strategy?
- Should you tell the truth about the investigation result?
- How can you use this information to help your team?
- Who should you share this information with?

REASONING: Process this information and plan how to use it strategically.
ACTION: Acknowledge investigation result
STATEMENT: What you tell other players about the investigation result (may be truthful or deceptive)"""
        
        return prompt
    
    def get_special_election_prompt(self, president_name: str, available_targets: List[str],
                                  game_state: Dict[str, Any], private_info: Dict[str, Any]) -> str:
        """Generate special election prompt."""
        
        prompt = f"""{self.base_instructions}

SPECIAL ELECTION
===============
President: {president_name}

CURRENT GAME STATE:
{self._format_game_state(game_state)}

YOUR PRIVATE INFORMATION:
{self._format_private_info(private_info)}

AVAILABLE CANDIDATES:
{', '.join(available_targets)}

You enacted a Fascist policy and gained the power to choose the next President.

Consider:
- Who do you trust to be President?
- Who would help your team's goals?
- How will this choice be perceived by others?
- Who might choose you as Chancellor if they become President?

REASONING: Analyze your options for the special election.
ACTION: Choose [Player Name] as next President
STATEMENT: Explain your choice to other players"""
        
        return prompt
    
    def get_execution_prompt(self, president_name: str, available_targets: List[str],
                           game_state: Dict[str, Any], private_info: Dict[str, Any]) -> str:
        """Generate execution target selection prompt."""
        
        prompt = f"""{self.base_instructions}

PRESIDENTIAL EXECUTION
=====================
President: {president_name}

CURRENT GAME STATE:
{self._format_game_state(game_state)}

YOUR PRIVATE INFORMATION:
{self._format_private_info(private_info)}

AVAILABLE EXECUTION TARGETS:
{', '.join(available_targets)}

You enacted a Fascist policy and gained the power to execute one player.
This permanently removes them from the game.

WARNING: If you execute Hitler, the Liberals immediately win!

Consider:
- Who is most dangerous to your team?
- Who might be Hitler?
- Who has been most suspicious?
- How will this execution be perceived?

REASONING: Analyze who to execute and the strategic implications.
ACTION: Execute [Player Name]
STATEMENT: Explain your execution choice to other players"""
        
        return prompt
    
    def get_nomination_announcement_prompt(self, president_name: str, chancellor_name: str,
                                         player_name: str, game_state: Dict[str, Any]) -> str:
        """Generate prompt for reacting to nomination announcement."""
        
        prompt = f"""{self.base_instructions}

NOMINATION ANNOUNCEMENT
======================
Player: {player_name}

CURRENT GAME STATE:
{self._format_game_state(game_state)}

NOMINATION:
{president_name} has nominated {chancellor_name} as Chancellor.

You will vote on this government soon. Consider:
- Do you trust this government?
- How does this nomination affect your strategy?
- What signals is the President sending?
- Should you try to influence other players' votes?

REASONING: Analyze the nomination and plan your response.
ACTION: Acknowledge nomination
STATEMENT: Your reaction to the nomination (may influence other players)"""
        
        return prompt
    
    def get_vote_results_prompt(self, player_name: str, vote_summary: Dict[str, Any],
                              individual_votes: List[tuple], game_state: Dict[str, Any]) -> str:
        """Generate prompt for reacting to vote results."""
        
        votes_text = "\n".join([f"{name}: {'Ja' if vote else 'Nein'}" for name, vote in individual_votes])
        
        prompt = f"""{self.base_instructions}

VOTE RESULTS
===========
Player: {player_name}

CURRENT GAME STATE:
{self._format_game_state(game_state)}

VOTING RESULTS:
Ja votes: {vote_summary['ja_votes']}
Nein votes: {vote_summary['nein_votes']}
Result: Government {vote_summary['result']}

INDIVIDUAL VOTES:
{votes_text}

Consider:
- How do these votes align with players' stated intentions?
- Who voted unexpectedly?
- What does this reveal about players' loyalties?
- How should you adjust your trust and strategy?

REASONING: Analyze the voting patterns and what they reveal about other players.
ACTION: Acknowledge vote results
STATEMENT: Your reaction to the vote results and any observations about voting patterns"""
        
        return prompt
    
    def _format_game_state(self, game_state: Dict[str, Any]) -> str:
        """Format game state for prompt inclusion."""
        return f"""
Phase: {game_state['phase']}
Players Alive: {', '.join(game_state['alive_players'])}
Liberal Policies: {game_state['liberal_policies']}/5
Fascist Policies: {game_state['fascist_policies']}/6
Election Failures: {game_state['election_failures']}/3
Current Government: President: {game_state['current_government']['president']}, Chancellor: {game_state['current_government']['chancellor']}
"""
    
    def _format_private_info(self, private_info: Dict[str, Any]) -> str:
        """Format private information for prompt inclusion."""
        info_lines = [f"Your Role: {private_info['your_role'].title()}"]
        
        if 'fascist_teammates' in private_info:
            info_lines.append(f"Fascist Teammates: {', '.join(private_info['fascist_teammates'])}")
        
        if 'hitler' in private_info:
            info_lines.append(f"Hitler: {private_info['hitler']}")
        
        if 'team_info' in private_info:
            info_lines.append(f"Team Info: {private_info['team_info']}")
        
        if 'last_investigation' in private_info:
            inv = private_info['last_investigation']
            info_lines.append(f"Last Investigation: {inv['target']} is {inv['result'].title()}")
        
        return "\n".join(info_lines)
    
    def _format_game_history(self, game_state: Dict[str, Any]) -> str:
        """Format game history for context."""
        history_lines = ["GAME HISTORY:"]
        
        # Policy history
        if 'policy_history' in game_state:
            history_lines.append("Past Policies Enacted:")
            for i, policy in enumerate(game_state['policy_history'], 1):
                history_lines.append(f"  Round {i}: {policy['type']} (President: {policy['president']}, Chancellor: {policy['chancellor']})")
        
        # Voting history
        if 'voting_history' in game_state:
            history_lines.append("\nRecent Voting Patterns:")
            for vote in game_state['voting_history'][-3:]:  # Last 3 votes
                history_lines.append(f"  {vote['president']} + {vote['chancellor']}: {vote['ja_votes']} Ja, {vote['nein_votes']} Nein - {vote['result']}")
        
        # Investigation results
        if 'investigation_history' in game_state:
            history_lines.append("\nInvestigation Claims:")
            for inv in game_state['investigation_history']:
                history_lines.append(f"  {inv['investigator']} claimed {inv['target']} is {inv['claimed_result']}")
        
        # Execution history
        if 'execution_history' in game_state:
            history_lines.append("\nExecutions:")
            for ex in game_state['execution_history']:
                history_lines.append(f"  {ex['executor']} executed {ex['target']} (was {ex['actual_role']})")
        
        # Election failures
        if game_state.get('election_failures', 0) > 0:
            history_lines.append(f"\nElection Failures: {game_state['election_failures']}/3")
        
        return "\n".join(history_lines)
    
    def _format_player_analysis(self, game_state: Dict[str, Any], current_player: str) -> str:
        """Format player behavior analysis."""
        analysis_lines = ["PLAYER BEHAVIOR ANALYSIS:"]
        
        if 'player_stats' in game_state:
            for player_id, stats in game_state['player_stats'].items():
                if player_id == current_player:
                    continue
                    
                player_name = stats.get('name', player_id)
                analysis_lines.append(f"\n{player_name}:")
                
                # Voting patterns
                if 'votes' in stats:
                    ja_votes = stats['votes'].count('ja')
                    total_votes = len(stats['votes'])
                    if total_votes > 0:
                        analysis_lines.append(f"  Voting: {ja_votes}/{total_votes} Ja votes ({ja_votes/total_votes*100:.0f}%)")
                
                # Government participation
                if 'governments' in stats:
                    analysis_lines.append(f"  Governments: {len(stats['governments'])} times in government")
                
                # Policy claims vs reality
                if 'policy_claims' in stats:
                    analysis_lines.append(f"  Policy Claims: {len(stats['policy_claims'])} claims made")
                
                # Suspicion level
                if 'suspicion_level' in stats:
                    analysis_lines.append(f"  Suspicion Level: {stats['suspicion_level']}/10")
        
        return "\n".join(analysis_lines)
    
    def _format_role_info(self, role_info: Dict[str, Any]) -> str:
        """Format role-specific information."""
        info_lines = [f"Your Role: {role_info['your_role'].title()}"]
        info_lines.append(f"Player Count: {role_info['player_count']}")
        info_lines.append(f"All Players: {', '.join(role_info['all_players'])}")
        
        if 'fascist_players' in role_info:
            info_lines.append(f"Fellow Fascists: {', '.join(role_info['fascist_players'])}")
        
        if 'hitler_player' in role_info:
            info_lines.append(f"Hitler: {role_info['hitler_player']}")
        
        if 'fascist_info' in role_info:
            info_lines.append(role_info['fascist_info'])
        
        return "\n".join(info_lines)
    
    def get_discussion_prompt(self, player_name: str, context: str, 
                            game_state: Dict[str, Any], private_info: Dict[str, Any]) -> str:
        """Generate prompt for general discussion/reaction."""
        
        prompt = f"""{self.base_instructions}

DISCUSSION
=========
Player: {player_name}
Context: {context}

CURRENT GAME STATE:
{self._format_game_state(game_state)}

YOUR PRIVATE INFORMATION:
{self._format_private_info(private_info)}

You may participate in discussion or remain silent. Consider:
- What information do you want to share or hide?
- How can you advance your team's goals?
- What suspicions should you voice or deflect?
- How can you build trust or sow discord?

REASONING: Plan your discussion strategy.
ACTION: Participate in discussion or remain silent
STATEMENT: What you say to other players (if anything)"""
        
        return prompt