"""Tests for DataAgent — does not require an Anthropic API key."""
import csv
from pathlib import Path

import pytest
from agents.base import AgentContext
from agents.data import DataAgent


@pytest.fixture
def csv_dataset(tmp_path: Path) -> Path:
    path = tmp_path / "tickets.csv"
    rows = [
        {"text": "My order has not arrived yet, where is it?", "label": "shipping"},
        {"text": "I want to cancel my subscription please", "label": "billing"},
        {"text": "The app crashes every time I open it", "label": "bug"},
        {"text": "How do I reset my password?", "label": "account"},
        {"text": "Got charged twice for the same item", "label": "billing"},
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.mark.asyncio
async def test_data_agent_profiles_csv(csv_dataset: Path):
    agent = DataAgent.__new__(DataAgent)
    ctx = AgentContext(
        run_id="test_run", user_intent="classify support tickets",
        dataset_path=str(csv_dataset),
        task_spec={"input_column": "text", "label_column": "label"},
    )
    result = await agent.run(ctx)
    assert result.success is True
    assert result.agent_name == "Data"
    assert result.next_agent == "Clean"  # DataAgent now routes through CleanAgent
    assert ctx.data_profile["num_rows"] == 5
    assert ctx.data_profile["num_classes"] == 4
    assert "billing" in ctx.data_profile["label_distribution"]


@pytest.mark.asyncio
async def test_data_agent_handles_missing_file():
    agent = DataAgent.__new__(DataAgent)
    ctx = AgentContext(run_id="test_run", user_intent="classify", dataset_path="/nonexistent/file.csv")
    result = await agent.run(ctx)
    assert result.success is False
    assert "not found" in result.message.lower()


@pytest.mark.asyncio
async def test_data_agent_flags_small_dataset(tmp_path: Path):
    path = tmp_path / "tiny.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerow({"text": "hello world", "label": "greet"})
    agent = DataAgent.__new__(DataAgent)
    ctx = AgentContext(run_id="test_run", user_intent="classify", dataset_path=str(path),
                       task_spec={"input_column": "text", "label_column": "label"})
    result = await agent.run(ctx)
    assert any("small" in issue.lower() for issue in ctx.data_profile["issues"])
