# Contributing to Web of Microbes

## Repository Structure

### GitHub (Code)
- **Repo**: `northenlab/webofmicrobes` (private)
- **Contents**: All source code, scripts, small data files

### HuggingFace (Large Files & Models)
- **LLM**: `northenlab/soilfm-qwen2.5-14b-literature-cpt` (public)
- **KG Data**: `northenlab/webofmicrobes-knowledge-graph` (private, dataset)
  - `soilfm_integrated_kg.pt` (71 MB)
  - `integrated_node_mappings.pkl` (54 MB)

## Git Workflow

We use branch-based development:

```
main     - stable, deployed to RunPod
dev      - active development
feature/ - specific features (optional)
```

### Daily Workflow

1. **Start work**: Make sure you're on dev branch
   ```bash
   git checkout dev
   git pull origin dev
   ```

2. **Make changes**: Edit files, test locally

3. **Commit**:
   ```bash
   git add .
   git commit -m "Brief description of changes"
   git push origin dev
   ```

4. **Deploy to production**: When dev is stable
   ```bash
   git checkout main
   git merge dev
   git push origin main
   ```

### Commit Message Format

Use clear, concise messages:
- `Add LLM input validation`
- `Fix Pseudomonas name resolution`
- `Update webapp UI for metabolite display`

## Local Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/northenlab/webofmicrobes.git
   cd webofmicrobes
   ```

2. Download KG files from HuggingFace:
   ```bash
   huggingface-cli download northenlab/webofmicrobes-knowledge-graph \
     --local-dir knowledge_graph/ --repo-type dataset
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Deployment (RunPod)

After merging to main, copy updated files to RunPod:
```bash
scp webapp/app.py root@<runpod-ip>:/workspace/webapp/
scp webapp/api.py root@<runpod-ip>:/workspace/webapp/
# etc.
```

Then restart the webapp on RunPod.

## Questions?

Contact the Northen Lab team.
