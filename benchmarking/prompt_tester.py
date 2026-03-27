import os
import glob
from dotenv import load_dotenv
from google import genai

load_dotenv()

PROMPTS_DIR = "prompts"
OUTPUT_DIR = "output"
NUM_RUNS = 5
API_KEY = os.getenv("GOOGLE_API_KEY")

MODELS = [
    "gemini-3.1-pro-preview",
    #"gemini-3-flash-preview",
]

print(MODELS)


client = genai.Client(api_key=API_KEY)

prompt_files = glob.glob(os.path.join(PROMPTS_DIR, "*.md"))

for prompt_path in prompt_files:
    prompt_name = os.path.splitext(os.path.basename(prompt_path))[0]
    with open(prompt_path, "r") as f:
        prompt_text = f.read().strip()

    for model in MODELS:
        model_tag = model.replace("/", "-")
        print(f"Prompt: {prompt_name} | Model: {model}")

        for run in range(1, NUM_RUNS + 1):
            print(f"  Run {run}/{NUM_RUNS}...", end=" ", flush=True)

            response = client.models.generate_content(
                model=model,
                contents=prompt_text,
                config=genai.types.GenerateContentConfig(temperature=0.1),
            )
            output_text = response.text

            output_filename = f"{prompt_name}_{model_tag}_run{run:02d}.md"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            with open(output_path, "w") as f:
                f.write(output_text)

            print("done")

print("All prompts completed.")
