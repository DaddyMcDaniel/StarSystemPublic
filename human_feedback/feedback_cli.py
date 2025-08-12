#!/usr/bin/env python3
"""
SUMMARY: Human Feedback CLI v1
===============================
Typer-based CLI tool for structured human feedback collection on StarSystem builds.
Validates input against human_feedback.v1.schema.json and saves results to runs/latest/.

KEY FEATURES:
- Interactive rubric scoring (1-5 scale) for visual quality, performance, usability
- Structured issue reporting with severity levels and evidence collection
- Recommendation tracking with priority and rationale
- Schema validation ensures data quality and consistency

USAGE:
  python human_feedback/feedback_cli.py collect
  python human_feedback/feedback_cli.py validate runs/latest/human_feedback.json
  python human_feedback/feedback_cli.py summary runs/latest/human_feedback.json

OUTPUT:
- Saves validated feedback to runs/latest/human_feedback.json
- Week 2 requirement for human-in-the-loop validation
- Supports iterative feedback collection and review workflows

RELATED FILES:
- schemas/human_feedback.v1.schema.json - Validation schema
- runs/{run_id}/human_feedback.json - Output artifacts
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

try:
    import typer
    import jsonschema
except ImportError:
    print("Missing dependencies. Install with: pip install typer jsonschema")
    exit(1)

app = typer.Typer(help="StarSystem Human Feedback Collection CLI")

# Schema path
SCHEMA_PATH = Path(__file__).parent.parent / "schemas" / "human_feedback.v1.schema.json"
RUNS_DIR = Path(__file__).parent.parent / "runs"
LATEST_DIR = RUNS_DIR / "latest"

def load_schema():
    """Load and return the human feedback schema for validation."""
    if not SCHEMA_PATH.exists():
        typer.echo(f"❌ Schema not found at {SCHEMA_PATH}", err=True)
        raise typer.Exit(1)
    
    with open(SCHEMA_PATH, 'r') as f:
        return json.load(f)

def validate_feedback(feedback_data: dict) -> tuple[bool, List[str]]:
    """Validate feedback against schema. Returns (is_valid, errors)."""
    try:
        schema = load_schema()
        jsonschema.validate(feedback_data, schema)
        return True, []
    except jsonschema.exceptions.ValidationError as e:
        return False, [str(e)]
    except Exception as e:
        return False, [f"Validation error: {e}"]

def collect_rubric_scores() -> dict:
    """Collect 1-5 rubric scores interactively."""
    typer.echo("\n📊 Rubric Scoring (1-5 scale, where 5 is excellent)")
    typer.echo("=" * 50)
    
    scores = {}
    rubric_items = [
        ("visual_quality", "Visual Quality & Aesthetics"),
        ("performance", "Performance & Responsiveness"),
        ("usability", "Usability & User Experience"),
        ("creativity_tools", "Creativity Tools & Features"),
        ("planet_generation", "Planet Generation Quality"),
        ("building_system", "Building System Functionality"),
        ("overall_experience", "Overall Experience")
    ]
    
    for key, description in rubric_items:
        while True:
            try:
                score = typer.prompt(f"{description} (1-5)")
                score = int(score)
                if 1 <= score <= 5:
                    scores[key] = score
                    break
                else:
                    typer.echo("❌ Score must be between 1 and 5")
            except ValueError:
                typer.echo("❌ Please enter a number between 1 and 5")
    
    return scores

def collect_issues() -> List[dict]:
    """Collect issue reports interactively."""
    typer.echo("\n🐛 Issue Reporting")
    typer.echo("=" * 20)
    
    issues = []
    
    while True:
        add_issue = typer.confirm("Add an issue?", default=False)
        if not add_issue:
            break
        
        issue = {
            "id": str(uuid.uuid4()),
            "severity": typer.prompt("Severity", type=typer.Choice(["critical", "major", "minor", "cosmetic"])),
            "category": typer.prompt("Category", type=typer.Choice(["performance", "visual", "usability", "functionality", "content"])),
            "description": typer.prompt("Description")
        }
        
        # Optional steps to reproduce
        if typer.confirm("Add reproduction steps?", default=False):
            steps = []
            typer.echo("Enter steps (empty line to finish):")
            while True:
                step = typer.prompt(f"Step {len(steps)+1}", default="")
                if not step:
                    break
                steps.append(step)
            
            if steps:
                issue["steps_to_reproduce"] = steps
        
        issues.append(issue)
        typer.echo(f"✅ Added {issue['severity']} {issue['category']} issue")
    
    return issues

def collect_recommendations() -> List[dict]:
    """Collect improvement recommendations."""
    typer.echo("\n💡 Recommendations")
    typer.echo("=" * 18)
    
    recommendations = []
    
    while True:
        add_rec = typer.confirm("Add a recommendation?", default=False)
        if not add_rec:
            break
        
        rec = {
            "priority": typer.prompt("Priority", type=typer.Choice(["high", "medium", "low"])),
            "area": typer.prompt("Area"),
            "suggestion": typer.prompt("Suggestion")
        }
        
        if typer.confirm("Add rationale?", default=False):
            rec["rationale"] = typer.prompt("Rationale")
        
        recommendations.append(rec)
        typer.echo(f"✅ Added {rec['priority']} priority recommendation")
    
    return recommendations

@app.command()
def collect(
    reviewer_id: str = typer.Option(..., "--reviewer", "-r", help="Reviewer identifier"),
    build_version: str = typer.Option("", "--build", "-b", help="Build version being reviewed"),
    platform: str = typer.Option("native", "--platform", "-p", help="Platform: web, native, or mobile")
):
    """Collect human feedback interactively and save to runs/latest/."""
    typer.echo("🎯 StarSystem Human Feedback Collection")
    typer.echo("=" * 40)
    
    # Ensure output directory exists
    LATEST_DIR.mkdir(parents=True, exist_ok=True)
    
    # Collect session metadata
    session_start = datetime.now(timezone.utc)
    
    # Collect feedback components
    rubric_scores = collect_rubric_scores()
    issues = collect_issues()
    recommendations = collect_recommendations()
    
    # Optional free-form comments
    typer.echo("\n💭 Additional Comments")
    comments = typer.prompt("Any additional comments?", default="", show_default=False)
    
    # Calculate session duration
    session_end = datetime.now(timezone.utc)
    duration_minutes = (session_end - session_start).total_seconds() / 60
    
    # Build feedback structure
    feedback = {
        "session_metadata": {
            "timestamp": session_start.isoformat(),
            "reviewer_id": reviewer_id,
            "session_duration_minutes": round(duration_minutes, 2),
            "platform": platform
        },
        "rubric_scores": rubric_scores,
        "issues": issues,
        "recommendations": recommendations
    }
    
    if build_version:
        feedback["session_metadata"]["build_version"] = build_version
    
    if comments.strip():
        feedback["free_form_comments"] = comments
    
    # Validate against schema
    is_valid, errors = validate_feedback(feedback)
    
    if not is_valid:
        typer.echo("❌ Validation failed:", err=True)
        for error in errors:
            typer.echo(f"  - {error}", err=True)
        raise typer.Exit(1)
    
    # Save feedback
    output_path = LATEST_DIR / "human_feedback.json"
    with open(output_path, 'w') as f:
        json.dump(feedback, f, indent=2)
    
    typer.echo(f"\n✅ Feedback saved to {output_path}")
    
    # Summary
    typer.echo("\n📋 Summary:")
    typer.echo(f"  - Rubric scores: {len(rubric_scores)} categories")
    typer.echo(f"  - Issues reported: {len(issues)}")
    typer.echo(f"  - Recommendations: {len(recommendations)}")
    typer.echo(f"  - Session duration: {duration_minutes:.1f} minutes")

@app.command()
def validate(
    feedback_file: Path = typer.Argument(..., help="Path to feedback JSON file")
):
    """Validate a feedback file against the schema."""
    if not feedback_file.exists():
        typer.echo(f"❌ File not found: {feedback_file}", err=True)
        raise typer.Exit(1)
    
    try:
        with open(feedback_file, 'r') as f:
            feedback_data = json.load(f)
    except json.JSONDecodeError as e:
        typer.echo(f"❌ Invalid JSON: {e}", err=True)
        raise typer.Exit(1)
    
    is_valid, errors = validate_feedback(feedback_data)
    
    if is_valid:
        typer.echo(f"✅ {feedback_file} is valid")
    else:
        typer.echo(f"❌ {feedback_file} validation failed:", err=True)
        for error in errors:
            typer.echo(f"  - {error}", err=True)
        raise typer.Exit(1)

@app.command()
def summary(
    feedback_file: Path = typer.Argument(..., help="Path to feedback JSON file")
):
    """Display a summary of feedback data."""
    if not feedback_file.exists():
        typer.echo(f"❌ File not found: {feedback_file}", err=True)
        raise typer.Exit(1)
    
    try:
        with open(feedback_file, 'r') as f:
            feedback = json.load(f)
    except json.JSONDecodeError as e:
        typer.echo(f"❌ Invalid JSON: {e}", err=True)
        raise typer.Exit(1)
    
    # Display summary
    typer.echo("📋 Feedback Summary")
    typer.echo("=" * 20)
    
    metadata = feedback.get("session_metadata", {})
    typer.echo(f"Reviewer: {metadata.get('reviewer_id', 'Unknown')}")
    typer.echo(f"Platform: {metadata.get('platform', 'Unknown')}")
    typer.echo(f"Duration: {metadata.get('session_duration_minutes', 0):.1f} minutes")
    
    # Rubric scores
    scores = feedback.get("rubric_scores", {})
    if scores:
        typer.echo("\n📊 Rubric Scores:")
        for category, score in scores.items():
            typer.echo(f"  {category.replace('_', ' ').title()}: {score}/5")
        
        avg_score = sum(scores.values()) / len(scores)
        typer.echo(f"  Average: {avg_score:.1f}/5")
    
    # Issues
    issues = feedback.get("issues", [])
    if issues:
        typer.echo(f"\n🐛 Issues: {len(issues)} total")
        severity_counts = {}
        for issue in issues:
            sev = issue.get("severity", "unknown")
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
        
        for severity, count in severity_counts.items():
            typer.echo(f"  {severity.title()}: {count}")
    
    # Recommendations
    recommendations = feedback.get("recommendations", [])
    if recommendations:
        typer.echo(f"\n💡 Recommendations: {len(recommendations)} total")
        priority_counts = {}
        for rec in recommendations:
            priority = rec.get("priority", "unknown")
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
        
        for priority, count in priority_counts.items():
            typer.echo(f"  {priority.title()}: {count}")
    
    # Comments
    if feedback.get("free_form_comments"):
        typer.echo(f"\n💭 Has additional comments: Yes")

if __name__ == "__main__":
    app()