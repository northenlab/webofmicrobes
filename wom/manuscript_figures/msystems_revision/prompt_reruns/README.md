# Re-running the SI agent prompts with the revised model

The revised model (BPR, `lambda_ord` = 0) is on `main` as of 88e99b3. The MCP server reads
`wom/trained_models/latest`, which resolves to `trained_models/ordinal_bpr_{consumed,secreted}`. The
models are copied into the `wom-mcp` image at build time, so the server must be rebuilt.

## 1. Redeploy (on the wom-chat.gnps2.org host)

```bash
git pull origin main
make server-compose-production   # docker compose build + up -d (both images)
```

## 2. Check that the site serves the revised model

| Query | Revised model (λ=0) | Submitted model (λ=0.1) | Plain BPR (`bpr_consumed`) |
|---|---|---|---|
| Acinetobacter, top NLDM carbon source | succinate, 10.91 | malate, 13.06 | pyruvate, 10.99 |
| Acinetobacter, spermidine | −1.46 | −3.76 | 1.94 |
| Pseudomonas, serine | 9.27 ± 1.76 | 10.13 ± 2.06 | 10.70 ± 1.86 |
| Pseudomonas, glycine | 5.45 ± 1.81 | 8.43 ± 2.19 | 8.63 ± 1.94 |

The submitted SI transcripts match the submitted model exactly, so the old site served that model, not plain BPR.
The server code has never read the `MODEL_DIR` variable that was in docker-compose.yml.

Each tool response also reports `model_dir`. It should end in `ordinal_bpr_consumed` (or `_secreted`).
The tools accept an optional `model_type`; if a transcript shows a different `model_dir`, re-run that prompt.

## 3. Prompts (exact user text from the SI; one fresh chat per prompt; Gemini Flash 3.0)

**Prompt 1**

> For Pseudomonas, rank all 23 of these substrates from highest to lowest preference: phenylalanine, glycine, threonine, isoleucine, leucine, lysine, serine, alanine, proline, arginine, glutamate, adenosine, guanine, hypoxanthine, uridine, 4-guanidinobutanoic acid, glucose, fructose, mannose, arabinose, trehalose, maltose, mannitol.

**Prompt 2** (two parts, entered one after the other in the same chat)

> What are the top 3 preferred substrates for these genera: Bacteroides, Blautia, Clostridium, Collinsella, Eggerthella, Eubacterium, Faecalibacterium, Prevotella?

> Can you make a non-redundant list of the metabolites in the above table and generate a table of genera vs the non-redundant metabolites showing BPR scores (no standard deviation)?

**Prompt 3**

> We want to dive deeper into the substrate utilization of Acinetobacter. We want to grow an Acinetobacter isolate on a sole carbon source that is present in Northen Lab Defined Medium (NLDM). Which carbon sources would you recommend (high BPR score) for this isolate? Provide a carbon source present in NLDM with a neutral score less than 5 and preferred by most microbes. Lastly, provide a low score (select the lowest score) to serve as controls. Take into account the solubility (at 50 mM in water) and toxicity (at 10-15 mM) for all selected carbon sources and If the best hits (i.e., lowest scoring) might not be a good fit based on solubility or toxicity, please suggest an alternative carbon source.

## 4. Save

Save each full response (with its tables) in this directory, e.g. `prompt1.md`, `prompt2_part1.md`,
`prompt2_part2.md`, `prompt3.md`, together with the date and the LLM used. The SI transcripts, the
Fig. S4 data sheets and panels, and the Prompt 3 note are then updated from these files.
