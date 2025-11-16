#!/usr/bin/env python3
"""
backend.py - Automated Documentation Update Backend Script (Gemini-only)

This script analyzes code changes, calls Google Gemini (via google-generativeai),
and produces a JSON output file for a GitHub Action to create PRs.

Usage (example):
    python backend.py --base-branch main --docs-path docs --output /tmp/auto_docs_result.json

IMPORTANT:
- This script expects a GitHub secret named `LLM_API_KEY` to be available as an
  environment variable (LLM_API_KEY). Do NOT hardcode your key.
- The script is Gemini-only (no OpenAI/Anthropic fallbacks).
"""

import os
import sys
import subprocess
import argparse
import json
import datetime
import re
from typing import List, Dict, Any
from pathlib import Path
import time

# Gemini SDK import (required)
try:
    import google.generativeai as genai  # type: ignore
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

# -----------------------------
# Utility helpers
# -----------------------------


def run_git_command(cmd: List[str], check: bool = True) -> str:
    """Run git command and return stdout (stripped)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=check,
            cwd=os.getcwd()
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        if check:
            print(f"❌ Git command failed: {' '.join(cmd)}")
            stderr = e.stderr or "<no stderr>"
            print(f"Error: {stderr.strip()}")
            raise
        return ""


# -----------------------------
# Git / Diff helpers
# -----------------------------


def get_changed_files(base_branch: str) -> List[str]:
    """
    Get a list of changed files between current HEAD and base_branch.
    Uses a robust '..' range that works in local and CI environments.
    """
    print(f"📋 Detecting changed files against base branch: {base_branch}")
    # Use double-dot range for robustness
    try:
        output = run_git_command(['git', 'diff', '--name-only', f'{base_branch}..HEAD'], check=False)
    except Exception:
        output = ""

    if not output:
        # As a last resort, try comparing HEAD to the base branch directly
        try:
            output = run_git_command(['git', 'ls-files', '--modified'], check=False)
        except Exception:
            output = ""

    if not output:
        print("⚠️  No changed files detected.")
        return []

    files = [line.strip() for line in output.splitlines() if line.strip()]

    # Filter out deleted/binary files
    binary_exts = {
        '.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico',
        '.pdf', '.zip', '.exe', '.dll', '.so', '.dylib',
        '.mp4', '.avi', '.mov', '.webm', '.woff', '.woff2',
        '.ttf', '.eot', '.otf'
    }
    existing_files = []
    for fpath in files:
        if any(fpath.lower().endswith(ext) for ext in binary_exts):
            print(f"⏭️  Skipping binary file: {fpath}")
            continue
        if os.path.exists(fpath):
            existing_files.append(fpath)
        else:
            print(f"⏭️  Skipping missing/deleted file: {fpath}")

    print(f"✅ Found {len(existing_files)} changed file(s).")
    return existing_files


def extract_diff(base_branch: str, max_chars: int = 15000) -> str:
    """
    Extract unified diff between current HEAD and base_branch.
    Truncates to max_chars to avoid huge prompts.
    """
    print(f"🔍 Extracting git diff against {base_branch}...")
    try:
        diff_text = run_git_command(['git', 'diff', f'{base_branch}..HEAD', '--unified=3'], check=False)
    except Exception:
        diff_text = ""

    if not diff_text:
        print("⚠️  No diff found.")
        return ""

    if len(diff_text) > max_chars:
        print(f"⚠️  Diff length {len(diff_text)} exceeds {max_chars} chars — truncating.")
        diff_text = diff_text[:max_chars] + "\n\n[... diff truncated ...]"

    snippet_len = min(len(diff_text), 1200)
    print(f"✅ Diff extracted ({len(diff_text)} chars). Showing first {snippet_len} chars:")
    print(diff_text[:snippet_len] + ("\n...[truncated preview]" if len(diff_text) > snippet_len else ""))
    return diff_text


# -----------------------------
# Documentation loading / matching
# -----------------------------


def load_related_docs(docs_path: str) -> Dict[str, str]:
    """
    Load markdown documentation files from docs_path.
    Returns mapping {relative_path: content}.
    """
    print(f"📚 Loading docs from: {docs_path}")
    if not docs_path:
        return {}
    if not os.path.exists(docs_path):
        print(f"⚠️  Docs path '{docs_path}' does not exist.")
        return {}

    docs = {}
    md_exts = {'.md', '.mdx', '.markdown'}
    ignore_dirs = {'node_modules', '.git', 'build', 'dist', '.next', '__pycache__', '.venv', 'venv', 'env'}
    for root, dirs, files in os.walk(docs_path):
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        for f in files:
            if any(f.lower().endswith(ext) for ext in md_exts):
                full = os.path.join(root, f)
                rel = os.path.relpath(full, start=os.getcwd())
                try:
                    with open(full, 'r', encoding='utf-8') as fh:
                        content = fh.read()
                    docs[rel] = content
                    print(f"   ✓ Loaded {rel} ({len(content)} chars)")
                except Exception as e:
                    print(f"   ⚠️ Failed to read {rel}: {e}")
    print(f"✅ Loaded {len(docs)} doc file(s).")
    return docs


def match_docs_to_changes(diff_text: str, docs_dict: Dict[str, str]) -> List[str]:
    """
    Heuristic matching: extract keywords from diff and search docs.
    Fallback: return all docs.
    """
    if not docs_dict:
        return []

    if not diff_text:
        return list(docs_dict.keys())

    print("🔗 Matching docs to code changes (heuristic)...")
    keywords = set()

    patterns = [
        r'def\s+(\w+)',
        r'class\s+(\w+)',
        r'function\s+(\w+)',
        r'const\s+(\w+)\s*=',
        r'export\s+(?:const|function|class)\s+(\w+)',
        r'<([A-Z][A-Za-z0-9_]+)(?:\s|>)'
    ]
    for pat in patterns:
        for m in re.findall(pat, diff_text):
            keywords.add(str(m).lower())

    quoted = re.findall(r'["\']([^"\']{3,80})["\']', diff_text)
    for q in quoted:
        keywords.add(q.lower().strip()[:60])

    print(f"   Keywords found: {len(keywords)}")
    matched = []
    for path, content in docs_dict.items():
        low = content.lower()
        for kw in keywords:
            if kw and kw in low:
                matched.append(path)
                break

    if not matched:
        print("   ⚠️ No specific matches — defaulting to all docs.")
        return list(docs_dict.keys())

    print(f"✅ Matched {len(matched)} doc file(s).")
    return matched


# -----------------------------
# Gemini integration (single provider)
# -----------------------------


def sanitize_text_for_json(s: str) -> str:
    """Remove zero-width and control characters likely to break JSON parsing."""
    # Remove zero-width spaces and common control characters
    s = s.replace('\u200b', '')
    s = s.replace('\u200c', '')
    s = s.replace('\u200d', '')
    # Optionally remove other weird unicode control chars
    s = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', s)
    return s


def call_llm_with_gemini(
    model_name: str,
    diff_text: str,
    relevant_docs_dict: Dict[str, str],
    api_key_env: str = "LLM_API_KEY",
    max_prompt_chars: int = 30000,
    max_output_tokens: int = 4096,
    retries: int = 3
) -> Dict[str, Any]:
    """
    Call Google Gemini to request doc updates.
    Returns dict with keys: updates (list), pr_title (str), pr_body (str).
    """
    if not GEMINI_AVAILABLE:
        raise RuntimeError("Gemini SDK (google-generativeai) is not installed in the environment.")

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"Environment variable '{api_key_env}' not set. Configure it in GitHub Secrets.")

    genai.configure(api_key=api_key)

    # Build truncated prompt contents to avoid excessive token usage
    truncated_diff = diff_text if len(diff_text) <= max_prompt_chars else diff_text[:max_prompt_chars] + "\n\n[... diff truncated ...]"

    # Prepare docs content with truncation per-file
    docs_combined = ""
    per_file_max = 12000  # conservative per-file chunk
    for path, content in relevant_docs_dict.items():
        safe_content = content if len(content) <= per_file_max else content[:per_file_max] + "\n\n[... content truncated ...]"
        docs_combined += f"\n\n===== FILE: {path} =====\n{safe_content}\n"

    prompt_header = (
        "You are an expert technical documentation rewriting AI.\n\n"
        "Task: Read the provided code/website diff and the existing documentation files. "
        "Update only those parts of the documentation that are impacted by the changes. "
        "Preserve headings, style and formatting. Return STRICT JSON ONLY (no explanation) with keys: "
        "updates, pr_title, pr_body.\n\n"
        "Structure of 'updates' items:\n"
        '[{"file": "relative/path/to/doc.md", "new_content": "FULL UPDATED FILE CONTENT"}]\n\n'
    )

    prompt = prompt_header + "\n\n## CODE/WEB DIFF:\n" + "```\n" + truncated_diff + "\n```\n\n" \
             + "## EXISTING DOCUMENTATION:\n" + docs_combined + "\n\n" \
             + "OUTPUT FORMAT (exact JSON):\n" \
             '{\n  "updates": [ { "file": "<path>", "new_content": "<full file content>" } ],\n' \
             '  "pr_title": "<short title>",\n' \
             '  "pr_body": "<detailed markdown body explaining changes>"\n}\n\n' \
             "IMPORTANT: Respond ONLY with the JSON object above and nothing else."

    # Safety: final prompt sanitization & size guard
    prompt = sanitize_text_for_json(prompt)
    if len(prompt) > max_prompt_chars:
        prompt = prompt[:max_prompt_chars] + "\n\n[... prompt truncated ...]"

    attempt = 0
    last_err = None
    while attempt < retries:
        attempt += 1
        try:
            print(f"🤖 Gemini request (model={model_name}) attempt {attempt}/{retries}...")
            model = genai.GenerativeModel(model_name)

            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.15,
                    "max_output_tokens": max_output_tokens,
                    "response_mime_type": "application/json"
                }
            )

            # The SDK returns .text containing the JSON (if response_mime_type applied)
            raw_text = getattr(response, "text", None)
            if raw_text is None:
                # Fallback: stringify response
                raw_text = str(response)

            raw_text = sanitize_text_for_json(raw_text).strip()

            # Remove surrounding code fences if any
            if raw_text.startswith("```") and raw_text.endswith("```"):
                # Strip the top and bottom fences
                lines = raw_text.splitlines()
                # remove first and last line usually the ``` markers
                raw_text = "\n".join(lines[1:-1]).strip()

            # Debug preview
            preview_len = min(len(raw_text), 800)
            print(f"✅ Received Gemini response ({len(raw_text)} chars). Preview:\n{raw_text[:preview_len]}{'...[truncated]' if len(raw_text)>preview_len else ''}")

            # Parse JSON
            parsed = json.loads(raw_text)

            # Validate required keys
            if not isinstance(parsed, dict):
                raise ValueError("Response JSON is not an object")
            for key in ("updates", "pr_title", "pr_body"):
                if key not in parsed:
                    raise ValueError(f"Missing required key in Gemini response: {key}")

            # Ensure updates is a list of dicts with 'file' and 'new_content'
            updates = parsed.get("updates") or []
            if not isinstance(updates, list):
                raise ValueError("'updates' must be a list")

            # Normalize entries
            normalized_updates = []
            for item in updates:
                if not isinstance(item, dict):
                    continue
                f = item.get("file")
                nc = item.get("new_content")
                if not f or not isinstance(nc, str):
                    continue
                # Normalize path: remove leading slashes/dots
                f_norm = os.path.normpath(f.lstrip("./"))
                normalized_updates.append({"file": f_norm, "new_content": nc})

            parsed["updates"] = normalized_updates
            # Truncate PR title if too long
            if isinstance(parsed.get("pr_title"), str):
                parsed["pr_title"] = parsed["pr_title"][:200]

            return parsed

        except Exception as e:
            last_err = e
            print(f"⚠️ Gemini attempt {attempt} failed: {e}")
            if attempt < retries:
                backoff = 2 ** attempt
                print(f"⏳ Retrying in {backoff}s...")
                time.sleep(backoff)
            else:
                break

    # If we reach here, all attempts failed
    raise RuntimeError(f"Gemini API failed after {retries} attempts. Last error: {last_err}")


# -----------------------------
# Apply updates & output writer
# -----------------------------


def apply_updates(updates: List[Dict[str, str]], dry_run: bool = False) -> List[str]:
    """
    Write updated documentation files to disk.
    Returns list of files updated (relative paths).
    """
    updated = []
    print(f"📝 Applying {len(updates)} update(s) (dry_run={dry_run})...")
    for up in updates:
        file_path = up.get("file")
        new_content = up.get("new_content", "")

        if not file_path:
            print("   ⚠️ Skipping update with missing file path")
            continue

        # Normalize path to be relative
        file_path = os.path.normpath(file_path.lstrip("./"))
        # Disallow absolute paths that escape repo root
        if os.path.isabs(file_path):
            # make it relative
            file_path = file_path.lstrip(os.sep)

        if dry_run:
            print(f"   [DRY RUN] Would update: {file_path} (content length: {len(new_content)} chars)")
            updated.append(file_path)
            continue

        # Ensure parent dir exists
        parent = os.path.dirname(file_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        try:
            with open(file_path, "w", encoding="utf-8") as fh:
                fh.write(new_content)
            print(f"   ✓ Wrote {file_path} ({len(new_content)} chars)")
            updated.append(file_path)
        except Exception as e:
            print(f"   ❌ Failed to write {file_path}: {e}")
            continue

    print(f"✅ Applied updates to {len(updated)} file(s).")
    return updated


def write_output(
    output_path: str,
    pr_branch: str,
    pr_title: str,
    pr_body: str,
    files_changed: List[str],
    created_changes: bool
):
    """Write the resulting JSON to output_path for GitHub Action to consume."""
    payload = {
        "pr_branch": pr_branch,
        "pr_title": pr_title,
        "pr_body": pr_body,
        "files_changed": files_changed,
        "created_changes": created_changes
    }
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"✅ Wrote action output to {output_path}")


# -----------------------------
# Main
# -----------------------------


def main():
    parser = argparse.ArgumentParser(description="Automated docs updater (Gemini-only)")
    parser.add_argument("--base-branch", type=str, default="main", help="Base branch to compare against")
    parser.add_argument("--docs-path", type=str, default="docs", help="Path to documentation folder")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSON file (for GitHub Action)")
    parser.add_argument("--llm-model", type=str, default="gemini-1.5-flash", help="Gemini model (gemini-*)")
    parser.add_argument("--max-chars", type=int, default=15000, help="Max chars for diff/docs passed to LLM")
    parser.add_argument("--dry-run", action="store_true", help="Do not write files; just simulate")
    args = parser.parse_args()

    # Validate Gemini availability
    if not GEMINI_AVAILABLE:
        print("❌ google-generativeai (Gemini SDK) not installed in this environment.")
        print("   Install it by adding to requirements.txt: google-generativeai")
        sys.exit(1)

    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        print("❌ Environment variable LLM_API_KEY not set. Add it to repository secrets in GitHub.")
        sys.exit(1)

    print("=" * 70)
    print("🚀 Automated Documentation Update (Gemini)")
    print("=" * 70)
    print(f"Base branch: {args.base_branch}")
    print(f"Docs path: {args.docs_path}")
    print(f"Output JSON: {args.output}")
    print(f"LLM model: {args.llm_model}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 70)

    try:
        # 1) Detect changed files
        changed_files = get_changed_files(args.base_branch)

        # 2) Extract diff
        diff_text = extract_diff(args.base_branch, max_chars=args.max_chars)
        if not diff_text:
            print("ℹ️  No diff detected; writing output and exiting.")
            timestamp = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
            write_output(
                args.output,
                pr_branch=f"auto-docs/update-{timestamp}",
                pr_title="📘 Auto-update docs: No changes detected",
                pr_body="No code changes were detected; no documentation updates required.",
                files_changed=[],
                created_changes=False
            )
            return

        # 3) Load docs
        all_docs = load_related_docs(args.docs_path)
        if not all_docs:
            print("⚠️  No docs found. Writing output and exiting.")
            timestamp = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
            write_output(
                args.output,
                pr_branch=f"auto-docs/update-{timestamp}",
                pr_title="📘 Auto-update docs: No documentation found",
                pr_body=f"No documentation files were found under '{args.docs_path}'.",
                files_changed=[],
                created_changes=False
            )
            return

        # 4) Match docs heuristically
        relevant_list = match_docs_to_changes(diff_text, all_docs)
        if not relevant_list:
            print("⚠️  No relevant docs matched. Exiting with no changes.")
            timestamp = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
            write_output(
                args.output,
                pr_branch=f"auto-docs/update-{timestamp}",
                pr_title="📘 Auto-update docs: No relevant docs matched",
                pr_body="Code changes detected but no documentation files appear relevant.",
                files_changed=[],
                created_changes=False
            )
            return

        relevant_docs = {p: all_docs[p] for p in relevant_list if p in all_docs}

        # 5) Call Gemini
        model_name = args.llm_model or "gemini-1.5-flash"
        if not model_name.startswith("gemini-"):
            print(f"⚠️ Provided model '{model_name}' does not look like a Gemini model. Defaulting to gemini-1.5-flash.")
            model_name = "gemini-1.5-flash"

        llm_result = call_llm_with_gemini(
            model_name=model_name,
            diff_text=diff_text,
            relevant_docs_dict=relevant_docs,
            api_key_env="LLM_API_KEY",
            max_prompt_chars=max(20000, args.max_chars)
        )

        # 6) Apply updates
        updates = llm_result.get("updates", [])
        updated_files = apply_updates(updates, dry_run=args.dry_run)

        # 7) Write output JSON for GitHub Action
        timestamp = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
        pr_branch = f"auto-docs/update-{timestamp}"
        pr_title = llm_result.get("pr_title", "📘 Auto-update docs")
        pr_body = llm_result.get("pr_body", "Automated documentation updates produced by LLM.")
        created_changes = len(updated_files) > 0

        write_output(
            args.output,
            pr_branch=pr_branch,
            pr_title=pr_title,
            pr_body=pr_body,
            files_changed=updated_files,
            created_changes=created_changes
        )

        # 8) Final status prints
        print("\n" + "=" * 70)
        if created_changes:
            print("✅ Documentation update completed.")
            print(f"PR Branch: {pr_branch}")
            print(f"PR Title: {pr_title}")
            print("Files updated:")
            for f in updated_files:
                print(f" - {f}")
        else:
            print("ℹ️ No documentation changes were created by the LLM (created_changes=False).")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user.")
        sys.exit(130)

    except Exception as exc:
        print("\n" + "=" * 70)
        print("❌ ERROR - Documentation update failed")
        print("=" * 70)
        print(f"Error: {exc}")
        print("Troubleshooting tips:")
        print(" - Ensure LLM_API_KEY is set in GitHub Secrets and passed in the workflow env.")
        print(" - Verify git checkout in GitHub Action uses fetch-depth: 0 (full history).")
        print(" - Confirm the base branch exists and is reachable.")
        print(" - Check API quota/usage limits for Gemini.")
        sys.exit(1)


if __name__ == "__main__":
    main()
