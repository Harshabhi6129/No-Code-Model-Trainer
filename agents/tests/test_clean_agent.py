"""Tests for CleanAgent."""
import csv
import asyncio
from pathlib import Path
import pytest

from agents.base import AgentContext
from agents.clean_agent import CleanAgent


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture()
def agent():
    return CleanAgent(client=None)


@pytest.fixture()
def ctx(tmp_path):
    return AgentContext(run_id="test", user_intent="classify", dataset_path=None,
                        task_spec={"input_column": "text", "label_column": "label"},
                        data_profile={"input_col": "text", "label_col": "label"})


@pytest.mark.asyncio
async def test_no_dataset(agent, ctx):
    result = await agent.run(ctx)
    assert result.success
    assert result.output["status"] == "skipped"


@pytest.mark.asyncio
async def test_removes_duplicates(agent, ctx, tmp_path):
    p = tmp_path / "data.csv"
    _write_csv(p, [
        {"text": "hello world", "label": "pos"},
        {"text": "hello world", "label": "pos"},  # duplicate
        {"text": "goodbye",     "label": "neg"},
    ])
    ctx.dataset_path = str(p)
    result = await agent.run(ctx)
    assert result.success
    assert result.output["removed_dups"] == 1
    assert result.output["cleaned_rows"] == 2


@pytest.mark.asyncio
async def test_removes_empty_text(agent, ctx, tmp_path):
    p = tmp_path / "data.csv"
    _write_csv(p, [
        {"text": "good text",  "label": "pos"},
        {"text": "",           "label": "neg"},   # empty
        {"text": "   ",        "label": "neg"},   # whitespace only
    ])
    ctx.dataset_path = str(p)
    result = await agent.run(ctx)
    assert result.success
    assert result.output["removed_nulls"] == 2
    assert result.output["cleaned_rows"] == 1


@pytest.mark.asyncio
async def test_clean_file_updates_context(agent, ctx, tmp_path):
    p = tmp_path / "data.csv"
    _write_csv(p, [
        {"text": "dup", "label": "a"},
        {"text": "dup", "label": "a"},
        {"text": "ok",  "label": "b"},
    ])
    ctx.dataset_path = str(p)
    await agent.run(ctx)
    assert ctx.dataset_path != str(p)
    assert "_cleaned" in ctx.dataset_path
    assert Path(ctx.dataset_path).exists()


@pytest.mark.asyncio
async def test_no_removal_keeps_original_path(agent, ctx, tmp_path):
    p = tmp_path / "data.csv"
    _write_csv(p, [
        {"text": "hello", "label": "pos"},
        {"text": "world", "label": "neg"},
    ])
    ctx.dataset_path = str(p)
    await agent.run(ctx)
    assert ctx.dataset_path == str(p)
