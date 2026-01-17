"""
evaluate.py - LLM-as-Judge Evaluation System

This script evaluates the EIP Specs Agent using an LLM-as-judge approach.
It reads interaction logs and uses a separate LLM to assess answer quality
based on multiple criteria.

Usage:
    python eval/evaluate.py --logs-dir app/logs --output eval/results.json

Evaluation Criteria:
    1. Relevance: Does the answer address the user's question?
    2. Accuracy: Is the information factually correct?
    3. Completeness: Does it cover all important aspects?
    4. Citations: Are sources properly referenced?
    5. Clarity: Is the answer well-structured and clear?

Output:
    - Per-question scores and feedback
    - Aggregate metrics (pass rate per criterion)
    - Summary statistics
"""

import os
import sys
import json
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from pydantic import BaseModel
from pydantic_ai import Agent


# ============================================================================
# EVALUATION SCHEMA
# ============================================================================

class EvaluationCheck(BaseModel):
    """Single evaluation criterion result."""
    name: str
    passed: bool
    reason: str


class EvaluationChecklist(BaseModel):
    """Complete evaluation result for one interaction."""
    checklist: list[EvaluationCheck]


# ============================================================================
# EVALUATION CRITERIA
# ============================================================================

EXPECTED_CHECKS = [
    "relevance",
    "accuracy",
    "completeness",
    "citations",
    "clarity"
]

EVALUATION_PROMPT = """
You are evaluating an AI agent that answers questions about the ethereum/EIPs repository.

Given a user question and the agent's answer, evaluate the response on these criteria:

1. **relevance**: Does the answer directly address what the user asked?
   - Pass if the answer is on-topic and attempts to answer the question
   - Fail if the answer is off-topic or doesn't address the question

2. **accuracy**: Is the information provided factually correct?
   - Pass if the technical details are accurate (based on your knowledge of EIPs)
   - Fail if there are factual errors or misleading information

3. **completeness**: Does the answer cover the key aspects of the question?
   - Pass if the main points are addressed adequately
   - Fail if major aspects are missing

4. **citations**: Does the answer include source references?
   - Pass if it mentions specific EIP numbers or file paths
   - Fail if no sources are cited when they should be

5. **clarity**: Is the answer well-structured and easy to understand?
   - Pass if the answer is clear and well-organized
   - Fail if it's confusing, rambling, or poorly structured

For each criterion, provide:
- name: the criterion name (lowercase)
- passed: true or false
- reason: brief explanation (1-2 sentences)

Be fair but rigorous. An answer can be good without being perfect.
""".strip()


# ============================================================================
# LOG PROCESSING
# ============================================================================

def load_log_files(logs_dir: str, agent_name: Optional[str] = None) -> list[dict]:
    """
    Load interaction logs from directory.

    Args:
        logs_dir: Path to logs directory
        agent_name: Optional filter for specific agent

    Returns:
        List of log record dictionaries
    """
    logs_path = Path(logs_dir)
    if not logs_path.exists():
        raise FileNotFoundError(f"Logs directory not found: {logs_dir}")

    log_files = sorted(logs_path.glob("*.json"), key=lambda p: p.stat().st_mtime)

    records = []
    for log_file in log_files:
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                record = json.load(f)

            # Filter by agent name if specified
            if agent_name and record.get("agent_name") != agent_name:
                continue

            record["_log_file"] = str(log_file)
            records.append(record)

        except Exception as e:
            print(f"Warning: Could not load {log_file}: {e}")

    return records


def extract_qa_from_log(log_record: dict) -> tuple[str, str]:
    """
    Extract question and answer from a log record.

    Args:
        log_record: Log dictionary with messages

    Returns:
        Tuple of (question, answer)
    """
    messages = log_record.get("messages", [])

    question = ""
    answer = ""

    for msg in messages:
        parts = msg.get("parts", [])
        for part in parts:
            if isinstance(part, dict):
                content = part.get("content", "")
            elif isinstance(part, str):
                content = part
            else:
                continue

            if msg.get("kind") == "request":
                question = content
            elif msg.get("kind") == "response":
                if isinstance(content, str) and content:
                    answer = content

    return question, answer


# ============================================================================
# EVALUATION LOGIC
# ============================================================================

async def evaluate_single_interaction(
    eval_agent: Agent,
    question: str,
    answer: str
) -> EvaluationChecklist:
    """
    Evaluate a single Q&A interaction.

    Args:
        eval_agent: PydanticAI agent configured for evaluation
        question: User's question
        answer: Agent's answer

    Returns:
        EvaluationChecklist with results for each criterion
    """
    prompt = f"""
## User Question
{question}

## Agent Answer
{answer}

Please evaluate this interaction according to the criteria.
"""

    result = await eval_agent.run(prompt, output_type=EvaluationChecklist)
    return result.output


async def run_evaluation(
    logs_dir: str,
    agent_name: Optional[str] = None,
    max_evals: int = 10,
    model: str = "gpt-4o-mini"
) -> dict:
    """
    Run full evaluation on logged interactions.

    Args:
        logs_dir: Path to logs directory
        agent_name: Optional filter for specific agent
        max_evals: Maximum number of interactions to evaluate
        model: LLM model for evaluation

    Returns:
        Dictionary with evaluation results and metrics
    """
    print(f"Loading logs from: {logs_dir}")
    records = load_log_files(logs_dir, agent_name)
    print(f"Found {len(records)} log records")

    if not records:
        return {"error": "No log records found", "results": []}

    # Limit number of evaluations
    eval_records = records[:max_evals]
    print(f"Evaluating {len(eval_records)} interactions")

    # Create evaluation agent
    eval_agent = Agent(
        name="eval_agent",
        model=model,
        instructions=EVALUATION_PROMPT,
        output_type=EvaluationChecklist,
    )

    # Run evaluations
    results = []
    for i, record in enumerate(eval_records):
        question, answer = extract_qa_from_log(record)

        if not question or not answer:
            print(f"  [{i+1}] Skipping - missing Q or A")
            continue

        print(f"  [{i+1}/{len(eval_records)}] Evaluating: {question[:50]}...")

        try:
            evaluation = await evaluate_single_interaction(eval_agent, question, answer)

            results.append({
                "log_file": record.get("_log_file", ""),
                "question": question,
                "answer": answer[:500] + "..." if len(answer) > 500 else answer,
                "checks": [c.model_dump() for c in evaluation.checklist]
            })
        except Exception as e:
            print(f"    Error: {e}")
            results.append({
                "log_file": record.get("_log_file", ""),
                "question": question,
                "error": str(e)
            })

    # Calculate aggregate metrics
    metrics = calculate_metrics(results)

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "logs_dir": logs_dir,
        "agent_name": agent_name or "all",
        "total_evaluated": len(results),
        "metrics": metrics,
        "results": results
    }


def calculate_metrics(results: list[dict]) -> dict:
    """
    Calculate aggregate metrics from evaluation results.

    Args:
        results: List of evaluation result dictionaries

    Returns:
        Dictionary with pass rates per criterion and overall
    """
    if not results:
        return {}

    # Count passes per criterion
    criterion_counts = {c: {"passed": 0, "total": 0} for c in EXPECTED_CHECKS}

    for result in results:
        if "error" in result:
            continue

        for check in result.get("checks", []):
            name = check.get("name", "").lower()
            if name in criterion_counts:
                criterion_counts[name]["total"] += 1
                if check.get("passed"):
                    criterion_counts[name]["passed"] += 1

    # Calculate pass rates
    metrics = {}
    total_passed = 0
    total_checks = 0

    for name, counts in criterion_counts.items():
        if counts["total"] > 0:
            rate = counts["passed"] / counts["total"]
            metrics[f"{name}_pass_rate"] = round(rate * 100, 1)
            total_passed += counts["passed"]
            total_checks += counts["total"]

    if total_checks > 0:
        metrics["overall_pass_rate"] = round(total_passed / total_checks * 100, 1)

    return metrics


def print_summary(evaluation_result: dict):
    """Print a formatted summary of evaluation results."""
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    print(f"\nAgent: {evaluation_result.get('agent_name', 'N/A')}")
    print(f"Interactions Evaluated: {evaluation_result.get('total_evaluated', 0)}")
    print(f"Timestamp: {evaluation_result.get('timestamp', 'N/A')}")

    metrics = evaluation_result.get("metrics", {})
    if metrics:
        print("\nPass Rates by Criterion:")
        print("-" * 40)
        for key, value in sorted(metrics.items()):
            if key != "overall_pass_rate":
                criterion = key.replace("_pass_rate", "").title()
                print(f"  {criterion:15} {value:5.1f}%")

        print("-" * 40)
        print(f"  {'Overall':15} {metrics.get('overall_pass_rate', 0):5.1f}%")

    print("\n" + "=" * 60)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate EIP Specs Agent using LLM-as-judge"
    )
    parser.add_argument(
        "--logs-dir", "-l",
        type=str,
        default="app/logs",
        help="Directory containing interaction logs (default: app/logs)"
    )
    parser.add_argument(
        "--agent-name", "-a",
        type=str,
        default=None,
        help="Filter logs by agent name (default: all)"
    )
    parser.add_argument(
        "--max-evals", "-n",
        type=int,
        default=10,
        help="Maximum interactions to evaluate (default: 10)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="eval/results.json",
        help="Output file for results (default: eval/results.json)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="gpt-4o-mini",
        help="LLM model for evaluation (default: gpt-4o-mini)"
    )

    args = parser.parse_args()

    # Run evaluation
    result = asyncio.run(run_evaluation(
        logs_dir=args.logs_dir,
        agent_name=args.agent_name,
        max_evals=args.max_evals,
        model=args.model
    ))

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {args.output}")

    # Print summary
    print_summary(result)


if __name__ == "__main__":
    main()
