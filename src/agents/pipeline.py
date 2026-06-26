"""
Lightweight multi-agent orchestration using Google Gemini API (free).
Each agent = system prompt + tools + Gemini API call.
Agents run sequentially, passing structured output to the next.
"""

import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable

import requests


GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent"

@dataclass
class Agent:
    """A single agent with a role, instructions, and tools."""
    name: str
    role: str
    instructions: str
    tools: dict[str, Callable] = field(default_factory=dict)

    def run(self, input_data: dict, verbose: bool = True) -> dict:
        """Execute this agent via Gemini API."""
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            return self._mock_run(input_data, verbose)

        prompt = f"""You are {self.name}, a {self.role}.

{self.instructions}

IMPORTANT: You must respond with a valid JSON object containing:
- "analysis": your reasoning and findings (string)
- "result": structured output data (object)
- "alerts": list of any alerts or warnings (list of strings)
- "status": "complete" or "needs_attention"

Respond with ONLY the JSON object. No markdown, no backticks, no preamble.

Input data:
{json.dumps(input_data, indent=2, default=str, ensure_ascii=False)}

Execute your analysis and respond with the JSON result."""

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 1500,
            },
        }

        start_time = time.time()

        try:
            response = requests.post(
                f"{GEMINI_URL}?key={api_key}",
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            return self._mock_run(input_data, verbose, error=str(e))

        elapsed = round(time.time() - start_time, 1)

        # Parse JSON response
        try:
            cleaned = raw_text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]
                cleaned = cleaned.rsplit("```", 1)[0]
            result = json.loads(cleaned)
        except json.JSONDecodeError:
            result = {
                "analysis": raw_text,
                "result": {},
                "alerts": ["Failed to parse structured output"],
                "status": "needs_attention",
            }

        if verbose:
            print(f"\n{'─'*60}")
            print(f"🤖 {self.name} ({elapsed}s)")
            print(f"{'─'*60}")
            analysis = result.get("analysis", "")
            if len(analysis) > 300:
                print(f"   {analysis[:300]}...")
            else:
                print(f"   {analysis}")
            alerts = result.get("alerts", [])
            if alerts:
                for alert in alerts:
                    print(f"   ⚠️  {alert}")
            print(f"   Status: {result.get('status', 'unknown')}")

        return {
            "agent": self.name,
            "output": result,
            "elapsed_seconds": elapsed,
            "raw_response": raw_text,
        }

    def _mock_run(self, input_data: dict, verbose: bool, error: str = "No API key") -> dict:
        """Fallback when API is unavailable."""
        if verbose:
            print(f"\n{'─'*60}")
            print(f"🤖 {self.name} (mock — {error})")
            print(f"{'─'*60}")

        return {
            "agent": self.name,
            "output": {
                "analysis": f"[Mock] {self.name} would analyze the input data here. ({error})",
                "result": {},
                "alerts": [f"Running in mock mode: {error}"],
                "status": "complete",
            },
            "elapsed_seconds": 0,
            "raw_response": "",
        }


class AgentPipeline:
    """Sequential pipeline of agents."""

    def __init__(self, agents: list[Agent], verbose: bool = True):
        self.agents = agents
        self.verbose = verbose
        self.execution_log: list[dict] = []

    def run(self, initial_input: dict) -> dict:
        context = {"initial_input": initial_input}
        self.execution_log = []

        if self.verbose:
            print(f"\n{'═'*60}")
            print(f"🚀 VayuDrishti Agent Pipeline — {len(self.agents)} agents")
            print(f"{'═'*60}")

        total_start = time.time()

        for agent in self.agents:
            result = agent.run(context, verbose=self.verbose)
            self.execution_log.append(result)
            context[agent.name] = result["output"]

        total_elapsed = round(time.time() - total_start, 1)

        if self.verbose:
            print(f"\n{'═'*60}")
            print(f"✅ Pipeline complete — {total_elapsed}s total")
            print(f"{'═'*60}")

        return {
            "final_output": context,
            "execution_log": self.execution_log,
            "total_elapsed": total_elapsed,
        }

    def get_log_for_display(self) -> list[dict]:
        display_log = []
        for entry in self.execution_log:
            display_log.append({
                "agent": entry["agent"],
                "time": f"{entry['elapsed_seconds']}s",
                "analysis": entry["output"].get("analysis", ""),
                "alerts": entry["output"].get("alerts", []),
                "status": entry["output"].get("status", "unknown"),
            })
        return display_log