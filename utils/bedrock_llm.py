"""
Bedrock LLM wrapper for DeepCode.

This module provides a custom LLM wrapper that uses AWS Bedrock to call Claude,
compatible with the mcp_agent AugmentedLLM interface.
"""

import os
from typing import Any, Dict, List, Optional, Type

# Check if we should use Bedrock
def is_bedrock_configured() -> bool:
    """Check if AWS Bedrock is configured."""
    aws_access_key = os.environ.get("AWS_ACCESS_KEY") or os.environ.get("AWS_ACCESS_KEY_ID", "")
    aws_secret_key = os.environ.get("AWS_SECRET_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    aws_region = os.environ.get("AWS_REGION", "")
    bedrock_model_arn = os.environ.get("ANTHROPIC_BEDROCK_MODEL_ARN", "")

    return bool(aws_access_key and aws_secret_key and aws_region and bedrock_model_arn)


def get_bedrock_config() -> Dict[str, str]:
    """Get Bedrock configuration from environment."""
    return {
        "aws_access_key": os.environ.get("AWS_ACCESS_KEY") or os.environ.get("AWS_ACCESS_KEY_ID", ""),
        "aws_secret_key": os.environ.get("AWS_SECRET_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
        "aws_region": os.environ.get("AWS_REGION", ""),
        "model_arn": os.environ.get("ANTHROPIC_BEDROCK_MODEL_ARN", ""),
    }


class BedrockAugmentedLLM:
    """
    A wrapper class that provides mcp_agent AugmentedLLM-compatible interface
    using AWS Bedrock for Claude API calls.
    """

    def __init__(self, agent: Any = None, **kwargs):
        """Initialize Bedrock LLM wrapper."""
        self.agent = agent
        self.config = get_bedrock_config()
        self._client = None
        self._initialized = False

    async def _ensure_client(self):
        """Lazily initialize the Bedrock client."""
        if not self._initialized:
            from anthropic import AsyncAnthropicBedrock

            self._client = AsyncAnthropicBedrock(
                aws_access_key=self.config["aws_access_key"],
                aws_secret_key=self.config["aws_secret_key"],
                aws_region=self.config["aws_region"],
            )
            self._initialized = True

    async def generate_str(
        self,
        message: str,
        request_params: Optional[Any] = None,
    ) -> str:
        """
        Generate a string response from the LLM.

        Args:
            message: The input message/prompt.
            request_params: Optional request parameters (maxTokens, temperature, etc.)

        Returns:
            The generated string response.
        """
        await self._ensure_client()

        # Extract parameters
        max_tokens = 8192
        temperature = 0.2
        system_prompt = ""

        if request_params:
            if hasattr(request_params, 'maxTokens'):
                max_tokens = request_params.maxTokens or max_tokens
            if hasattr(request_params, 'max_tokens'):
                max_tokens = request_params.max_tokens or max_tokens
            if hasattr(request_params, 'temperature'):
                temperature = request_params.temperature if request_params.temperature is not None else temperature
            if hasattr(request_params, 'systemPrompt'):
                system_prompt = request_params.systemPrompt or ""
            if hasattr(request_params, 'system_prompt'):
                system_prompt = request_params.system_prompt or ""

        # Build messages
        messages = [{"role": "user", "content": message}]

        try:
            # Make API call
            kwargs = {
                "model": self.config["model_arn"],
                "max_tokens": max_tokens,
                "messages": messages,
                "temperature": temperature,
            }

            if system_prompt:
                kwargs["system"] = system_prompt

            response = await self._client.messages.create(**kwargs)

            # Extract text from response
            result = ""
            for block in response.content:
                if hasattr(block, 'text'):
                    result += block.text

            return result

        except Exception as e:
            print(f"Bedrock API error: {e}")
            raise

    async def generate(
        self,
        message: str,
        request_params: Optional[Any] = None,
    ) -> Any:
        """
        Generate a response from the LLM (returns full response object).

        Args:
            message: The input message/prompt.
            request_params: Optional request parameters.

        Returns:
            The response object.
        """
        await self._ensure_client()

        max_tokens = 8192
        temperature = 0.2
        system_prompt = ""

        if request_params:
            if hasattr(request_params, 'maxTokens'):
                max_tokens = request_params.maxTokens or max_tokens
            if hasattr(request_params, 'max_tokens'):
                max_tokens = request_params.max_tokens or max_tokens
            if hasattr(request_params, 'temperature'):
                temperature = request_params.temperature if request_params.temperature is not None else temperature
            if hasattr(request_params, 'systemPrompt'):
                system_prompt = request_params.systemPrompt or ""

        messages = [{"role": "user", "content": message}]

        kwargs = {
            "model": self.config["model_arn"],
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": temperature,
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        return await self._client.messages.create(**kwargs)


def get_bedrock_llm_class() -> Type[BedrockAugmentedLLM]:
    """
    Return the Bedrock LLM class.

    This is used as a factory function compatible with mcp_agent's attach_llm().
    """
    return BedrockAugmentedLLM


def get_llm_class_with_bedrock_fallback() -> Type[Any]:
    """
    Get the appropriate LLM class, preferring Bedrock if configured.

    Returns:
        The LLM class to use (BedrockAugmentedLLM or standard mcp_agent LLM).
    """
    if is_bedrock_configured():
        print("🤖 Using BedrockAugmentedLLM (AWS Bedrock configuration found)")
        return BedrockAugmentedLLM

    # Fall back to standard mcp_agent LLM selection
    from utils.llm_utils import get_preferred_llm_class
    return get_preferred_llm_class()
