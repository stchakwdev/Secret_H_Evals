"""
OpenRouter client for LLM API calls with cost tracking and rate limiting.
"""
import asyncio
import json
import time
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import aiohttp
import os
from datetime import datetime, timedelta

from config.openrouter_config import (
    OPENROUTER_CONFIG, 
    OPENROUTER_MODELS,
    get_model_for_decision,
    get_model_config,
    estimate_cost,
    COST_LIMITS
)

@dataclass
class APIRequest:
    """Track API request details."""
    timestamp: datetime
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost: float
    latency: float
    decision_type: str
    player_id: str
    success: bool
    error: Optional[str] = None

@dataclass
class APIResponse:
    """Structured API response."""
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost: float
    latency: float
    success: bool
    error: Optional[str] = None

class CostTracker:
    """Track API costs and enforce limits."""
    
    def __init__(self):
        self.requests: List[APIRequest] = []
        self.current_game_cost = 0.0
        self.daily_cost = 0.0
        self.tournament_cost = 0.0
        self.last_reset = datetime.now().date()
    
    def add_request(self, request: APIRequest):
        """Add a request to tracking."""
        self.requests.append(request)
        self.current_game_cost += request.cost
        self.daily_cost += request.cost
        self.tournament_cost += request.cost
        
        # Reset daily cost if new day
        today = datetime.now().date()
        if today > self.last_reset:
            self.daily_cost = request.cost
            self.last_reset = today
    
    def check_limits(self) -> Tuple[bool, str]:
        """Check if we're within cost limits."""
        if self.current_game_cost > COST_LIMITS['per_game']:
            return False, f"Game cost limit exceeded: ${self.current_game_cost:.2f}"
        
        if self.daily_cost > COST_LIMITS['daily_limit']:
            return False, f"Daily cost limit exceeded: ${self.daily_cost:.2f}"
        
        if self.tournament_cost > COST_LIMITS['per_tournament']:
            return False, f"Tournament cost limit exceeded: ${self.tournament_cost:.2f}"
        
        return True, ""
    
    def should_alert(self) -> bool:
        """Check if we should send cost alert."""
        for threshold in COST_LIMITS['alert_thresholds']:
            if (self.current_game_cost >= threshold and 
                not any(r.cost >= threshold for r in self.requests[:-1])):
                return True
        return False
    
    def reset_game_cost(self):
        """Reset game-specific cost tracking."""
        self.current_game_cost = 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cost and usage statistics."""
        if not self.requests:
            return {}
        
        return {
            'total_requests': len(self.requests),
            'total_cost': sum(r.cost for r in self.requests),
            'avg_cost_per_request': sum(r.cost for r in self.requests) / len(self.requests),
            'avg_latency': sum(r.latency for r in self.requests) / len(self.requests),
            'success_rate': sum(1 for r in self.requests if r.success) / len(self.requests),
            'current_game_cost': self.current_game_cost,
            'daily_cost': self.daily_cost,
            'tournament_cost': self.tournament_cost,
            'model_usage': self._get_model_usage()
        }
    
    def _get_model_usage(self) -> Dict[str, int]:
        """Get usage count by model."""
        usage = {}
        for request in self.requests:
            usage[request.model] = usage.get(request.model, 0) + 1
        return usage

class OpenRouterClient:
    """Async client for OpenRouter API with intelligent model routing."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session: Optional[aiohttp.ClientSession] = None
        self.cost_tracker = CostTracker()
        self.rate_limiter = asyncio.Semaphore(10)  # Max 10 concurrent requests
        
    async def __aenter__(self):
        # Create SSL context that allows unverified certificates (for testing)
        import ssl
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=OPENROUTER_CONFIG['timeout']),
            connector=connector,
            headers={
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
                **OPENROUTER_CONFIG['headers']
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def make_request(
        self,
        prompt: str,
        decision_type: str,
        player_id: str,
        model_override: Optional[str] = None,
        temperature_override: Optional[float] = None
    ) -> APIResponse:
        """Make an API request with automatic model selection and cost tracking."""
        
        # Check cost limits before making request
        within_limits, limit_msg = self.cost_tracker.check_limits()
        if not within_limits:
            return APIResponse(
                content="",
                model="",
                prompt_tokens=0,
                completion_tokens=0,
                cost=0.0,
                latency=0.0,
                success=False,
                error=f"Cost limit exceeded: {limit_msg}"
            )
        
        # Select model
        model_name = model_override or get_model_for_decision(decision_type)
        model_config = get_model_config(model_name)
        
        if not model_config:
            return APIResponse(
                content="",
                model=model_name,
                prompt_tokens=0,
                completion_tokens=0,
                cost=0.0,
                latency=0.0,
                success=False,
                error=f"Unknown model: {model_name}"
            )
        
        # Prepare request
        temperature = temperature_override or model_config.temperature
        request_data = {
            "model": model_config.name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": model_config.max_tokens,
            "temperature": temperature,
            "stream": False
        }
        
        start_time = time.time()
        
        async with self.rate_limiter:
            try:
                for attempt in range(OPENROUTER_CONFIG['max_retries']):
                    try:
                        async with self.session.post(
                            f"{OPENROUTER_CONFIG['base_url']}/chat/completions",
                            json=request_data
                        ) as response:
                            
                            latency = time.time() - start_time
                            
                            if response.status == 200:
                                data = await self._parse_response_json(response)
                                if data is not None:
                                    # Validate response structure
                                    if self._validate_response_structure(data):
                                        return await self._process_success_response(
                                            data, model_config.name, decision_type, player_id, latency
                                        )
                                    else:
                                        # Try to proceed with potentially malformed response
                                        return await self._process_success_response(
                                            data, model_config.name, decision_type, player_id, latency
                                        )
                                else:
                                    return APIResponse(
                                        content="",
                                        model=model_config.name,
                                        prompt_tokens=0,
                                        completion_tokens=0,
                                        cost=0.0,
                                        latency=latency,
                                        success=False,
                                        error="Failed to parse JSON response"
                                    )
                            
                            elif response.status == 429:  # Rate limited
                                if attempt < OPENROUTER_CONFIG['max_retries'] - 1:
                                    wait_time = 2 ** attempt  # Exponential backoff
                                    await asyncio.sleep(wait_time)
                                    continue
                                else:
                                    return await self._handle_rate_limit(
                                        model_config.name, decision_type, player_id, latency
                                    )
                            
                            else:
                                error_text = await response.text()
                                return APIResponse(
                                    content="",
                                    model=model_config.name,
                                    prompt_tokens=0,
                                    completion_tokens=0,
                                    cost=0.0,
                                    latency=latency,
                                    success=False,
                                    error=f"HTTP {response.status}: {error_text}"
                                )
                    
                    except asyncio.TimeoutError:
                        if attempt < OPENROUTER_CONFIG['max_retries'] - 1:
                            continue
                        
                        return APIResponse(
                            content="",
                            model=model_config.name,
                            prompt_tokens=0,
                            completion_tokens=0,
                            cost=0.0,
                            latency=time.time() - start_time,
                            success=False,
                            error="Request timeout"
                        )
                    
                    except Exception as e:
                        if attempt < OPENROUTER_CONFIG['max_retries'] - 1:
                            continue
                        
                        return APIResponse(
                            content="",
                            model=model_config.name,
                            prompt_tokens=0,
                            completion_tokens=0,
                            cost=0.0,
                            latency=time.time() - start_time,
                            success=False,
                            error=str(e)
                        )
                
            except Exception as e:
                return APIResponse(
                    content="",
                    model=model_config.name,
                    prompt_tokens=0,
                    completion_tokens=0,
                    cost=0.0,
                    latency=time.time() - start_time,
                    success=False,
                    error=str(e)
                )
    
    async def _process_success_response(
        self, 
        data: Dict,
        model_name: str,
        decision_type: str,
        player_id: str,
        latency: float
    ) -> APIResponse:
        """Process successful API response with enhanced parsing and fallbacks."""
        
        # Extract content with multiple fallback strategies
        content, extraction_error = self._extract_content_with_fallbacks(data)
        
        # Extract usage with fallbacks and estimation
        prompt_tokens, completion_tokens, usage_error = self._extract_usage_with_fallbacks(
            data, content, model_name
        )
        
        # Calculate cost
        cost = estimate_cost(prompt_tokens, completion_tokens, model_name)
        
        # Determine success status
        success = content is not None and len(content.strip()) > 0
        error_msg = None
        
        if not success:
            error_details = []
            if extraction_error:
                error_details.append(f"Content: {extraction_error}")
            if usage_error:
                error_details.append(f"Usage: {usage_error}")
            error_msg = "; ".join(error_details) if error_details else "Unknown parsing error"
        
        # Track the request
        request = APIRequest(
            timestamp=datetime.now(),
            model=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=cost,
            latency=latency,
            decision_type=decision_type,
            player_id=player_id,
            success=success,
            error=error_msg
        )
        self.cost_tracker.add_request(request)
        
        # Check for cost alerts
        if self.cost_tracker.should_alert():
            print(f"⚠️  Cost alert: Game cost ${self.cost_tracker.current_game_cost:.2f}")
        
        return APIResponse(
            content=content or "",
            model=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=cost,
            latency=latency,
            success=success,
            error=error_msg
        )
    
    def _extract_content_with_fallbacks(self, data: Dict) -> Tuple[Optional[str], Optional[str]]:
        """Extract content with multiple fallback strategies."""
        
        # Handle None or empty data gracefully
        if not data or not isinstance(data, dict):
            return None, "Invalid or empty data structure"
        
        # Strategy 1: Standard OpenAI format
        try:
            if 'choices' in data and len(data['choices']) > 0:
                choice = data['choices'][0]
                if 'message' in choice and 'content' in choice['message']:
                    content = choice['message']['content']
                    if content is not None:
                        return content, None
        except (KeyError, IndexError, TypeError) as e:
            pass
        
        # Strategy 2: Alternative choice access patterns
        try:
            # Try different choice access patterns
            for choice_path in [
                ['choices', 0, 'message', 'content'],
                ['choices', 0, 'text'],
                ['choices', 0, 'content'],
                ['response', 'content'],
                ['output', 'content'],
                ['result', 'content'],
                ['text'],
                ['content']
            ]:
                try:
                    current = data
                    for key in choice_path:
                        if isinstance(current, list) and isinstance(key, int):
                            current = current[key]
                        elif isinstance(current, dict) and key in current:
                            current = current[key]
                        else:
                            break
                    else:
                        # Successfully navigated the path
                        if isinstance(current, str) and current.strip():
                            return current, None
                except (KeyError, IndexError, TypeError):
                    continue
        except Exception as e:
            pass
        
        # Strategy 3: Raw response fallback
        try:
            if isinstance(data, dict):
                # Look for any string values that might be content
                for key, value in data.items():
                    if isinstance(value, str) and len(value) > 10 and key.lower() in ['content', 'text', 'response', 'output', 'result']:
                        return value, None
        except Exception as e:
            pass
        
        # Strategy 4: Convert entire response to string as last resort
        try:
            if data:
                fallback_content = str(data)
                if len(fallback_content) > 50:  # Reasonable length threshold
                    return fallback_content, "Used raw response conversion"
        except Exception as e:
            pass
        
        return None, f"Could not extract content from response structure: {list(data.keys()) if isinstance(data, dict) else type(data)}"
    
    def _extract_usage_with_fallbacks(self, data: Dict, content: Optional[str], model_name: str) -> Tuple[int, int, Optional[str]]:
        """Extract token usage with fallbacks and estimation."""
        
        # Strategy 1: Standard usage field
        try:
            usage = data.get('usage', {})
            if usage:
                prompt_tokens = usage.get('prompt_tokens', 0)
                completion_tokens = usage.get('completion_tokens', 0)
                
                # Validate reasonable values
                if prompt_tokens >= 0 and completion_tokens >= 0:
                    return prompt_tokens, completion_tokens, None
        except Exception as e:
            pass
        
        # Strategy 2: Alternative usage field locations
        try:
            for usage_path in [
                ['usage'],
                ['token_usage'],
                ['tokens'],
                ['billing', 'usage'],
                ['meta', 'usage'],
                ['statistics', 'tokens']
            ]:
                try:
                    current = data
                    for key in usage_path:
                        current = current[key]
                    
                    if isinstance(current, dict):
                        prompt_tokens = current.get('prompt_tokens', current.get('input_tokens', 0))
                        completion_tokens = current.get('completion_tokens', current.get('output_tokens', 0))
                        
                        if prompt_tokens >= 0 and completion_tokens >= 0:
                            return prompt_tokens, completion_tokens, None
                except (KeyError, TypeError):
                    continue
        except Exception as e:
            pass
        
        # Strategy 3: Estimate based on content length
        try:
            if content:
                # Rough estimation: ~4 characters per token
                estimated_completion_tokens = max(1, len(content) // 4)
                # Assume prompt was similar length (very rough)
                estimated_prompt_tokens = estimated_completion_tokens
                
                return estimated_prompt_tokens, estimated_completion_tokens, "Estimated from content length"
        except Exception as e:
            pass
        
        # Strategy 4: Use model defaults
        try:
            model_config = get_model_config(model_name)
            if model_config:
                # Very conservative estimates
                default_prompt = 100
                default_completion = 50
                return default_prompt, default_completion, "Used model defaults"
        except Exception as e:
            pass
        
        # Fallback: Minimal values
        return 10, 10, "Used minimal fallback values"
    
    async def _parse_response_json(self, response: aiohttp.ClientResponse) -> Optional[Dict]:
        """Parse JSON response with multiple fallback strategies."""
        
        # Strategy 1: Standard JSON parsing
        try:
            return await response.json()
        except Exception as e:
            pass
        
        # Get text for fallback strategies
        try:
            text = await response.text()
        except Exception:
            return None
        
        if not text or not text.strip():
            return None
        
        # Strategy 2: Parse as text first, then JSON
        try:
            # Try to clean up common JSON issues
            cleaned_text = self._clean_json_text(text)
            return json.loads(cleaned_text)
        except Exception as e:
            pass
        
        # Strategy 3: Extract JSON from mixed content
        try:
            # Look for JSON-like structures in the text
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
            if json_match:
                potential_json = json_match.group(0)
                return json.loads(potential_json)
        except Exception as e:
            pass
        
        # Strategy 4: Try different JSON extraction patterns
        try:
            # Look for more complex nested JSON
            nested_json_match = re.search(r'\{.*?"choices".*?\}', text, re.DOTALL)
            if nested_json_match:
                potential_json = nested_json_match.group(0)
                return json.loads(potential_json)
        except Exception as e:
            pass
        
        # Strategy 5: Create minimal structure from text
        try:
            if text.strip():
                return {
                    "choices": [{"message": {"content": text.strip()}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": max(1, len(text) // 4)}
                }
        except Exception as e:
            pass
        
        return None
    
    def _clean_json_text(self, text: str) -> str:
        """Clean up common JSON formatting issues."""
        # Remove BOM and other invisible characters
        text = text.strip('\ufeff\ubbef')
        
        # Fix common JSON issues
        text = text.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
        
        # Fix unescaped quotes (basic attempt)
        text = re.sub(r'(?<!\\)"(?=(?:[^"\\]|\\.)+")', '\\"', text)
        
        return text
    
    def _validate_response_structure(self, data: Dict) -> bool:
        """Validate that response has expected structure."""
        try:
            # Check for basic OpenAI-compatible structure
            if isinstance(data, dict):
                # Look for choices array or content field
                if 'choices' in data:
                    if isinstance(data['choices'], list) and len(data['choices']) > 0:
                        choice = data['choices'][0]
                        if isinstance(choice, dict):
                            return True
                
                # Look for direct content fields
                if any(key in data for key in ['content', 'text', 'response', 'output']):
                    return True
                
                # Any dictionary with string values might be valid
                if any(isinstance(v, str) for v in data.values()):
                    return True
        except Exception:
            pass
        
        return False
    
    async def _handle_rate_limit(
        self,
        model_name: str,
        decision_type: str,
        player_id: str,
        latency: float
    ) -> APIResponse:
        """Handle rate limiting with model fallback."""
        if not OPENROUTER_CONFIG['model_fallbacks']:
            return APIResponse(
                content="",
                model=model_name,
                prompt_tokens=0,
                completion_tokens=0,
                cost=0.0,
                latency=latency,
                success=False,
                error="Rate limited - no fallback enabled"
            )
        
        # Try cheaper fallback models
        current_model_config = get_model_config(model_name)
        if not current_model_config:
            return APIResponse(
                content="",
                model=model_name,
                prompt_tokens=0,
                completion_tokens=0,
                cost=0.0,
                latency=latency,
                success=False,
                error="Rate limited - no fallback available"
            )
        
        # Find cheaper models in same or lower tier
        fallback_models = []
        for config in OPENROUTER_MODELS.values():
            if (config.cost_per_1k_tokens < current_model_config.cost_per_1k_tokens and
                config.name != model_name):
                fallback_models.append(config.name)
        
        if fallback_models:
            # Try the first fallback model
            print(f"⚠️  Rate limited for {model_name}, trying fallback: {fallback_models[0]}")
            return await self.make_request(
                prompt="", decision_type=decision_type, player_id=player_id,
                model_override=fallback_models[0]
            )
        
        return APIResponse(
            content="",
            model=model_name,
            prompt_tokens=0,
            completion_tokens=0,
            cost=0.0,
            latency=latency,
            success=False,
            error="Rate limited - no suitable fallback"
        )
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get comprehensive cost and usage summary."""
        return self.cost_tracker.get_stats()
    
    def reset_game_tracking(self):
        """Reset game-specific cost tracking."""
        self.cost_tracker.reset_game_cost()
    
    def export_usage_log(self, filename: str):
        """Export usage log to JSON file."""
        with open(filename, 'w') as f:
            json.dump([asdict(req) for req in self.cost_tracker.requests], f, 
                     indent=2, default=str)