from __future__ import annotations

from typing import Any

import pandas as pd

from src.hf_client import HFClient


def _hf_json(system_prompt: str, user_payload: dict[str, Any]) -> dict[str, Any]:
    client = HFClient()
    return client.generate_json(system_prompt=system_prompt, user_payload=user_payload)


def _fallback_agentic_reasoning(strategy: str, error_message: str) -> dict[str, Any]:
    return {
        "diagnosis": {
            "root_causes": ["Automated reasoning service unavailable for this model endpoint."],
            "capability_gaps": ["Detailed LLM-based diagnosis could not be generated."],
            "risk_level": "Medium",
            "diagnostic_summary": (
                f"Fallback diagnosis used for strategy '{strategy[:120]}'."
            ),
        },
        "proposal": {
            "recommended_actions": [
                "Clarify action ownership and accountable teams.",
                "Add milestone-based execution checkpoints.",
            ],
            "recommended_kpis": [
                "Define baseline and target for each strategy KPI.",
                "Track monthly progress and variance to target.",
            ],
            "timeline_scope_adjustments": [
                "Break broad actions into phased deliverables.",
                "Prioritize high-impact actions mapped directly to strategy outcomes.",
            ],
            "expected_alignment_delta": 0.0,
            "proposal_summary": "Fallback proposal generated due to unavailable LLM endpoint.",
        },
        "critique": {
            "approved": False,
            "issues": ["LLM endpoint unavailable; using deterministic fallback recommendations."],
            "confidence": 0.2,
        },
        "generation_mode": "fallback",
        "generation_error": error_message,
    }


def _agentic_reasoning(strategy: str, matched_actions: str, similarity_score: float) -> dict[str, Any]:
    system_prompt = (
        "You are an agentic strategic planner using diagnose-propose-critique reasoning. "
        "Return strict JSON with keys diagnosis, proposal, critique."
    )
    payload = {
        "task": "Perform diagnose-propose-critique reasoning for strategic alignment.",
        "strategy": strategy,
        "matched_actions": matched_actions,
        "similarity_score": similarity_score,
        "required_output_schema": {
            "diagnosis": {
                "root_causes": ["string"],
                "capability_gaps": ["string"],
                "risk_level": "Low|Medium|High",
                "diagnostic_summary": "string",
            },
            "proposal": {
                "recommended_actions": ["string"],
                "recommended_kpis": ["string"],
                "timeline_scope_adjustments": ["string"],
                "expected_alignment_delta": "float",
                "proposal_summary": "string",
            },
            "critique": {
                "approved": "boolean",
                "issues": ["string"],
                "confidence": "float",
            },
        },
    }
    return _hf_json(system_prompt, payload)


def run_agentic_reasoning_layer(
    low_alignment_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if low_alignment_df.empty:
        return low_alignment_df.copy(), []

    recommendation_rows: list[dict[str, Any]] = []
    reasoning_log: list[dict[str, Any]] = []

    for row in low_alignment_df.itertuples(index=False):
        strategy = str(row.strategy)
        matched_actions = str(row.matched_actions)
        similarity_score = float(row.similarity_score)
        strategy_chunk_id = str(getattr(row, "strategy_chunk_id", ""))

        try:
            reasoning_result = _agentic_reasoning(strategy, matched_actions, similarity_score)
        except Exception as exc:
            reasoning_result = _fallback_agentic_reasoning(strategy=strategy, error_message=str(exc))
        diagnosis = reasoning_result.get("diagnosis", {})
        proposal = reasoning_result.get("proposal", {})
        critique = reasoning_result.get("critique", {})

        recommendation_rows.append(
            {
                "strategy_chunk_id": strategy_chunk_id,
                "strategy": strategy,
                "baseline_similarity_score": similarity_score,
                "recommended_actions": " || ".join(proposal.get("recommended_actions", [])),
                "recommended_kpis": " || ".join(proposal.get("recommended_kpis", [])),
                "timeline_scope_adjustments": " || ".join(
                    proposal.get("timeline_scope_adjustments", [])
                ),
                "expected_alignment_delta": float(proposal.get("expected_alignment_delta", 0.0)),
                "critic_approved": bool(critique.get("approved", False)),
                "critic_confidence": float(critique.get("confidence", 0.0)),
            }
        )

        reasoning_log.append(
            {
                "strategy_chunk_id": strategy_chunk_id,
                "strategy": strategy,
                "diagnosis": diagnosis,
                "final_proposal": proposal,
                "final_critique": critique,
            }
        )

    recommendations_df = pd.DataFrame(recommendation_rows)
    return recommendations_df, reasoning_log
