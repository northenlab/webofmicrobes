import os
import re
import glob
from collections import defaultdict
from dotenv import load_dotenv
from google import genai

load_dotenv()

OUTPUT_DIR = "output"
PROMPTS_DIR = "prompts"
SUMMARY_DIR = "summary"
API_KEY = os.getenv("GOOGLE_API_KEY")
SUMMARY_MODEL = "gemini-3.1-pro-preview"

client = genai.Client(api_key=API_KEY)

os.makedirs(SUMMARY_DIR, exist_ok=True)

# Group files by (model_tag, prompt_name)
# Filename format: {prompt_name}_{model_tag}_run{NN}.md
groups = defaultdict(lambda: defaultdict(list))
for filepath in sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.md"))):
    filename = os.path.basename(filepath)
    match = re.match(r'^(.+?)_(.+)_run(\d+)\.md$', filename)
    if not match:
        continue
    prompt_name, model_tag, run_num = match.group(1), match.group(2), int(match.group(3))
    groups[model_tag][prompt_name].append((run_num, filepath))

for model_tag, prompts in sorted(groups.items()):
    print(f"\nModel: {model_tag}")

    output_filename = f"{model_tag}_summary.md"
    output_path = os.path.join(SUMMARY_DIR, output_filename)

    with open(output_path, "w") as out:
        out.write(f"# Summary: {model_tag}\n\n")
        out.write(f"**Prompts analyzed:** {len(prompts)}\n\n")
        out.write("---\n\n")

        for prompt_name, runs in sorted(prompts.items()):
            runs.sort()
            print(f"  Summarizing: {prompt_name} ({len(runs)} runs)...")

            # Load original prompt text
            prompt_path = os.path.join(PROMPTS_DIR, f"{prompt_name}.md")
            if os.path.exists(prompt_path):
                with open(prompt_path) as f:
                    prompt_text = f.read().strip()
            else:
                prompt_text = "(prompt file not found)"

            # Load all trial outputs
            trial_texts = []
            for run_num, filepath in runs:
                with open(filepath) as f:
                    trial_texts.append(f"## Trial {run_num}\n{f.read().strip()}")
            combined = "\n\n---\n\n".join(trial_texts)

            summary_prompt = f"""You are analyzing {len(runs)} independent trials of the same prompt run through an LLM.

Your tasks:
1. Identify the most common or consensus answer across all trials.
2. Summarize the top/best answer concisely.
3. Highlight any key differences or inconsistencies between trials.
4. Rate consistency on a scale from 1-10 (10 = perfectly consistent across all trials).

Here are the {len(runs)} trial outputs:

{combined}
"""

            response = client.models.generate_content(
                model=SUMMARY_MODEL,
                contents=summary_prompt,
                config=genai.types.GenerateContentConfig(temperature=0.1),
            )

            out.write(f"## Prompt: {prompt_name}\n\n")
            out.write(f"> {prompt_text}\n\n")
            out.write(f"**Trials analyzed:** {len(runs)}\n\n")
            out.write(response.text)
            out.write("\n\n---\n\n")

    print(f"  Saved -> {output_path}")

print("\nAll summaries completed.")
