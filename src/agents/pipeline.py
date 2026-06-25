"""
Lightweight multi-agent orchestration.
Replaces CrewAI with direct Anthropic SDK calls — same agent architecture,
zero dependency bloat (no kubernetes, no chromadb).

Each agent = system prompt + tools + Claude API call.
Agents run sequentially, passing structured output to the next.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Callable

import anthropic


@dataclass
class Agent:
    """A single agent with a role, instructions, and tools."""
    name: str
    role: str
    instructions: str
    tools: dict[str, Callable] = field(default_factory=dict)

    def run(self, input_data: dict, verbose: bool = True) -> dict:
        """
        Execute this agent: call Claude with role + input, optionally use tools.

        Args:
            input_data: structured data from previous agent or initial input
            verbose: print agent reasoning to console

        Returns:
            dict with 'output' (structured result) and 'reasoning' (text log)
        """
        client = anthropic.Anthropic()

        system_prompt = f"""You are {self.name}, a {self.role}.

{self.instructions}

IMPORTANT: You must respond with a valid JSON object containing:
- "analysis": your reasoning and findings (string)
- "result": structured output data (object)
- "alerts": list of any alerts or warnings (list of strings)
- "status": "complete" or "needs_attention"

Respond with ONLY the JSON object. No markdown, no backticks, no preamble."""

        tool_descriptions = ""
        if self.tools:
            tool_descriptions = "\n\nAvailable tools (already executed, results in input):\n"
            for name, func in self.tools.items():
                tool_descriptions += f"- {name}: {func.__doc__ or 'No description'}\n"

        user_message = f"""Input data:
{json.dumps(input_data, indent=2, default=str, ensure_ascii=False)}
{tool_descriptions}
Execute your analysis and respond with the JSON result."""

        start_time = time.time()

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1500,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )

        elapsed = round(time.time() - start_time, 1)
        raw_text = response.content[0].text

        # Parse JSON response
        try:
            # Strip any markdown fencing
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


class AgentPipeline:
    """
    Sequential pipeline of agents.
    Each agent receives the accumulated context from all previous agents.
    """

    def __init__(self, agents: list[Agent], verbose: bool = True):
        self.agents = agents
        self.verbose = verbose
        self.execution_log: list[dict] = []

    def run(self, initial_input: dict) -> dict:
        """
        Execute all agents in sequence.

        Returns:
            dict with final output and full execution log.
        """
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

            # Add this agent's output to context for the next agent
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
        """
        Get a display-friendly version of the execution log
        for the Streamlit 'Agent Activity' page.
        """
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
