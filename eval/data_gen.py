"""
data_gen.py - Evaluation Dataset Generator

This script generates question-answer pairs from EIP documents for evaluating
the agent's performance. It uses an LLM to create realistic questions that
users might ask about Ethereum Improvement Proposals.

Usage:
    python eval/data_gen.py --num-questions 50 --output eval/questions.json

The generated dataset contains:
    - question: A natural language question about EIPs
    - expected_answer: Key points the answer should contain
    - source_file: The EIP document the question is based on
    - category: Type of question (factual, conceptual, comparison, etc.)
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from openai import OpenAI
from ingest import read_repo_data


# Question generation prompt template
QUESTION_GEN_PROMPT = """
Based on the following EIP document content, generate {num_questions} diverse questions
that a user might ask about this EIP. Include a mix of:

1. Factual questions (What, When, Who)
2. Conceptual questions (How, Why)
3. Comparison questions (vs other EIPs/standards)
4. Implementation questions (technical details)

For each question, provide:
- The question itself
- Key points that a correct answer should include
- The category of question

Document Title: {title}
Document Path: {path}
Content:
{content}

Respond in JSON format:
[
  {{
    "question": "...",
    "expected_points": ["point1", "point2", ...],
    "category": "factual|conceptual|comparison|implementation"
  }},
  ...
]
"""


def generate_questions_for_doc(
    client: OpenAI,
    doc: dict,
    num_questions: int = 3,
    model: str = "gpt-4o-mini"
) -> list[dict]:
    """
    Generate evaluation questions from a single EIP document.

    Args:
        client: OpenAI client instance
        doc: Document dictionary with content and metadata
        num_questions: Number of questions to generate per document
        model: LLM model to use for generation

    Returns:
        List of question dictionaries with source metadata
    """
    # Truncate content if too long
    content = doc.get("content", "")[:4000]
    title = doc.get("title", doc.get("path", "Unknown"))
    path = doc.get("path", "")

    prompt = QUESTION_GEN_PROMPT.format(
        num_questions=num_questions,
        title=title,
        path=path,
        content=content
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates evaluation questions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)
        questions = result if isinstance(result, list) else result.get("questions", [])

        # Add source metadata to each question
        for q in questions:
            q["source_file"] = path
            q["source_title"] = title
            q["eip_number"] = doc.get("eip", "")

        return questions

    except Exception as e:
        print(f"Error generating questions for {path}: {e}")
        return []


def generate_evaluation_dataset(
    num_questions: int = 50,
    questions_per_doc: int = 3,
    repo_owner: str = "ethereum",
    repo_name: str = "EIPs",
    output_path: str = "eval/questions.json",
    model: str = "gpt-4o-mini"
) -> list[dict]:
    """
    Generate a complete evaluation dataset from repository documents.

    Args:
        num_questions: Total number of questions to generate
        questions_per_doc: Questions to generate per document
        repo_owner: GitHub repository owner
        repo_name: Repository name
        output_path: Path to save the generated dataset
        model: LLM model for question generation

    Returns:
        List of all generated questions
    """
    print(f"Downloading repository: {repo_owner}/{repo_name}...")
    docs, branch = read_repo_data(repo_owner, repo_name)
    print(f"Found {len(docs)} documents from {branch} branch")

    # Filter to documents with substantial content
    docs = [d for d in docs if len(d.get("content", "")) > 500]
    print(f"Filtered to {len(docs)} documents with sufficient content")

    # Calculate how many docs to sample
    num_docs = min(len(docs), (num_questions + questions_per_doc - 1) // questions_per_doc)
    sampled_docs = random.sample(docs, num_docs)
    print(f"Sampling {num_docs} documents for question generation")

    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    all_questions = []
    for i, doc in enumerate(sampled_docs):
        print(f"Processing document {i+1}/{num_docs}: {doc.get('path', 'unknown')}")
        questions = generate_questions_for_doc(
            client, doc, questions_per_doc, model
        )
        all_questions.extend(questions)

        if len(all_questions) >= num_questions:
            break

    # Trim to exact count
    all_questions = all_questions[:num_questions]

    # Save to file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_questions, f, indent=2, ensure_ascii=False)

    print(f"\nGenerated {len(all_questions)} questions")
    print(f"Saved to: {output_path}")

    # Print category distribution
    categories = {}
    for q in all_questions:
        cat = q.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

    print("\nCategory distribution:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")

    return all_questions


def main():
    parser = argparse.ArgumentParser(
        description="Generate evaluation questions from EIP documents"
    )
    parser.add_argument(
        "--num-questions", "-n",
        type=int,
        default=50,
        help="Total number of questions to generate (default: 50)"
    )
    parser.add_argument(
        "--questions-per-doc", "-q",
        type=int,
        default=3,
        help="Questions to generate per document (default: 3)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="eval/questions.json",
        help="Output file path (default: eval/questions.json)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="gpt-4o-mini",
        help="LLM model for generation (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--repo-owner",
        type=str,
        default="ethereum",
        help="GitHub repository owner (default: ethereum)"
    )
    parser.add_argument(
        "--repo-name",
        type=str,
        default="EIPs",
        help="GitHub repository name (default: EIPs)"
    )

    args = parser.parse_args()

    generate_evaluation_dataset(
        num_questions=args.num_questions,
        questions_per_doc=args.questions_per_doc,
        repo_owner=args.repo_owner,
        repo_name=args.repo_name,
        output_path=args.output,
        model=args.model
    )


if __name__ == "__main__":
    main()
