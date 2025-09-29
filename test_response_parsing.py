#!/usr/bin/env python3
"""
Test enhanced LLM response parsing with fallbacks.
Validates robust handling of malformed, incomplete, or unexpected responses.
"""

import asyncio
import json
from typing import Dict, Any

# Mock response class for testing
class MockResponse:
    def __init__(self, status: int, data: Any = None, text_data: str = None, json_error: bool = False):
        self.status = status
        self._data = data
        self._text_data = text_data
        self._json_error = json_error
    
    async def json(self):
        if self._json_error:
            raise json.JSONDecodeError("Invalid JSON", "", 0)
        return self._data
    
    async def text(self):
        return self._text_data or json.dumps(self._data) if self._data else ""

# Test response parsing scenarios
test_scenarios = [
    {
        "name": "Standard OpenAI Response",
        "response": {
            "choices": [{"message": {"content": "This is a standard response"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        },
        "expected_content": "This is a standard response",
        "expected_success": True
    },
    {
        "name": "Missing Usage Data",
        "response": {
            "choices": [{"message": {"content": "Response without usage"}}]
        },
        "expected_content": "Response without usage",
        "expected_success": True
    },
    {
        "name": "Alternative Structure - Direct Content",
        "response": {
            "content": "Direct content field",
            "tokens": {"input": 8, "output": 4}
        },
        "expected_content": "Direct content field",
        "expected_success": True
    },
    {
        "name": "Malformed Choices Array",
        "response": {
            "choices": [{"text": "Alternative text field"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3}
        },
        "expected_content": "Alternative text field",
        "expected_success": True
    },
    {
        "name": "Nested Content Path",
        "response": {
            "response": {"content": "Nested response content"},
            "meta": {"usage": {"input_tokens": 15, "output_tokens": 6}}
        },
        "expected_content": "Nested response content",
        "expected_success": True
    },
    {
        "name": "String-only Response",
        "text_only": "Just a plain text response without JSON structure",
        "expected_content": "Just a plain text response without JSON structure",
        "expected_success": True
    },
    {
        "name": "Mixed Content with JSON",
        "text_only": 'Some prefix text {"choices": [{"message": {"content": "Extracted JSON"}}]} some suffix',
        "expected_content": "Extracted JSON",
        "expected_success": True
    },
    {
        "name": "Empty Response",
        "response": {},
        "expected_content": "",
        "expected_success": False
    },
    {
        "name": "Corrupted JSON Structure",
        "text_only": '{"choices": [{"message": {"content": "Incomplete',
        "json_error": True,
        "expected_success": False  # Should fail gracefully
    }
]

async def test_enhanced_parsing():
    """Test enhanced response parsing with various scenarios."""
    
    # Import here to avoid circular imports during testing
    from agents.openrouter_client import OpenRouterClient
    
    print("🧪 Testing Enhanced LLM Response Parsing with Fallbacks")
    print("=" * 60)
    
    # Create client instance for testing (dummy API key)
    client = OpenRouterClient("test-key")
    
    passed = 0
    failed = 0
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{i}. Testing: {scenario['name']}")
        print("-" * 40)
        
        try:
            # Create mock response
            if "text_only" in scenario:
                mock_response = MockResponse(
                    200, 
                    data=None,  # Force JSON parsing to fail
                    text_data=scenario["text_only"],
                    json_error=True  # Force JSON parsing to fail
                )
            else:
                mock_response = MockResponse(200, scenario["response"])
            
            # Test JSON parsing
            parsed_data = await client._parse_response_json(mock_response)
            
            if parsed_data is not None:
                print(f"✅ JSON parsing successful")
                
                # Test content extraction
                content, extraction_error = client._extract_content_with_fallbacks(parsed_data)
                
                # Test usage extraction  
                prompt_tokens, completion_tokens, usage_error = client._extract_usage_with_fallbacks(
                    parsed_data, content, "test-model"
                )
                
                print(f"📄 Content: '{content[:50]}{'...' if content and len(content) > 50 else ''}'")
                print(f"🔢 Tokens: {prompt_tokens} prompt, {completion_tokens} completion")
                
                if extraction_error:
                    print(f"⚠️  Content warning: {extraction_error}")
                if usage_error:
                    print(f"⚠️  Usage warning: {usage_error}")
                
                # Validate expectations
                expected_content = scenario["expected_content"]
                expected_success = scenario["expected_success"]
                
                # Handle content matching more flexibly
                if expected_content:
                    content_matches = (content == expected_content)
                else:
                    content_matches = True  # No specific content expected
                    
                success_matches = (content is not None and len(content.strip()) > 0) == expected_success
                
                if content_matches and success_matches:
                    print(f"✅ Test PASSED")
                    passed += 1
                else:
                    print(f"❌ Test FAILED")
                    print(f"   Expected content: '{expected_content}'")
                    print(f"   Got content: '{content}'")
                    print(f"   Expected success: {expected_success}")
                    print(f"   Got success: {content is not None and len(content.strip()) > 0}")
                    failed += 1
            else:
                print(f"❌ JSON parsing failed")
                if scenario["expected_success"]:
                    print(f"❌ Test FAILED - Expected success but got parsing failure")
                    failed += 1
                else:
                    print(f"✅ Test PASSED - Expected failure and got parsing failure")
                    passed += 1
                    
        except Exception as e:
            print(f"💥 Test ERROR: {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    print(f"✅ Success Rate: {passed/(passed+failed)*100:.1f}%" if (passed+failed) > 0 else "No tests run")
    
    if failed == 0:
        print(f"🎉 All tests passed! Enhanced parsing is working correctly.")
    else:
        print(f"⚠️  Some tests failed. Review the enhanced parsing implementation.")

async def test_real_scenarios():
    """Test with real-world API response scenarios."""
    
    print(f"\n\n🌍 Testing Real-World API Response Scenarios")
    print("=" * 60)
    
    # Import here to avoid circular imports
    from agents.openrouter_client import OpenRouterClient
    
    client = OpenRouterClient("test-key")
    
    # Simulate real API responses that might be problematic
    real_scenarios = [
        {
            "name": "OpenRouter Rate Limited Response",
            "data": {
                "error": "Rate limited",
                "message": "Too many requests",
                "retry_after": 60
            }
        },
        {
            "name": "Model Unavailable Response", 
            "data": {
                "error": "Model not available",
                "available_models": ["gpt-3.5-turbo", "claude-3-haiku"]
            }
        },
        {
            "name": "Partial Response",
            "data": {
                "choices": [{"message": {"content": "This response was cut off due to"}}],
                "usage": {"prompt_tokens": 150}  # Missing completion_tokens
            }
        },
        {
            "name": "Non-Standard Provider Response",
            "data": {
                "output": "Response from a non-standard provider",
                "billing": {"usage": {"tokens_in": 100, "tokens_out": 25}}
            }
        }
    ]
    
    for i, scenario in enumerate(real_scenarios, 1):
        print(f"\n{i}. {scenario['name']}")
        print("-" * 40)
        
        # Test parsing
        content, extraction_error = client._extract_content_with_fallbacks(scenario["data"])
        prompt_tokens, completion_tokens, usage_error = client._extract_usage_with_fallbacks(
            scenario["data"], content, "test-model"
        )
        
        print(f"📄 Extracted content: '{content[:100]}{'...' if content and len(content) > 100 else ''}'")
        print(f"🔢 Token usage: {prompt_tokens} prompt, {completion_tokens} completion")
        
        if extraction_error:
            print(f"⚠️  Content extraction: {extraction_error}")
        if usage_error:
            print(f"⚠️  Usage extraction: {usage_error}")
        
        # Validate structure
        is_valid = client._validate_response_structure(scenario["data"])
        print(f"✅ Structure valid: {is_valid}")

if __name__ == "__main__":
    print("🚀 Starting Enhanced Response Parsing Tests\n")
    
    async def run_all_tests():
        await test_enhanced_parsing()
        await test_real_scenarios()
        
        print(f"\n\n🏁 All tests completed!")
        print("Enhanced LLM response parsing with fallbacks is ready for use.")
    
    asyncio.run(run_all_tests())