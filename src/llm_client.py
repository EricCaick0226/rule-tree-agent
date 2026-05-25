class MockLLMClient:
    def generate(self, prompt: str) -> str:
        """Offline placeholder for future LLM integration.

        Future versions can replace this with OpenAI-compatible API calls.
        Even with a real LLM, every generated category, grade, description,
        and rule must remain grounded in retrieved document evidence.
        """
        return ""

