# Smoke test: SI agent prompts with the revised model

`scripts/revision_prompt_smoke_test.py` replays the tool calls that the WoM agent made in the runs shown in
the SI through the MCP server's own tool functions, once with the submitted model (λ=0.1) and once with the
revised model (λ=0). No LLM is called, so the new wording an LLM writes from these tool outputs may differ.

Runs used: Prompt 1 = chat `24ec6051`, Prompt 2 = `defb77d6`, Prompt 3 = `3fba3c05`. Prompt 3 was chosen
from 12 runs by text similarity to the SI transcript (0.94; the next best was 0.43).

**Validation:** the submitted-model replay reproduces the transcripts and the current Supplementary Fig. 4.
- Quoted scores agree to within 0.01: Prompt 1 (22 values), Prompt 2 (8), Prompt 3 (6).
- Panel a reproduces exactly (16/1/4/1, r = 0.605, 90.9%).
- Panel b reproduces (r = −0.627; ρ −0.662 vs −0.658 printed).
- Panel c reproduces exactly (29/2/30/16, r = 0.555, 76.6%).

## Prompt 1 (Pseudomonas, 22 substrates): small change in conclusions; every number changes

- 20 of 22 scores fall, by 0.5–3.2 points; maltose rises by 2.1 and 4-guanidinobutanoic acid by 0.4. Spearman ρ between the old and new rankings is 0.90.
- The top 2 are unchanged (serine, glutamate).
- Several substrates move down: glycine 9 → 16, adenosine 4 → 9, uridine 12 → 17, guanine 3 → 7. Lysine moves up, 8 → 4.
- **Supplementary Fig. 4a:** the confusion matrix is unchanged (16/1/4/1, 90.9%), and r improves from 0.605 to 0.680.
  - Uridine is still the one false positive, but its score falls from 7.77 to 4.53, close to the threshold of 4.0.
- **Supplementary Fig. 4b:** the correlation is stronger. r goes from −0.627 to −0.712, and ρ from −0.66 to −0.70.

## Prompt 2 (8 gut genera, top 3): moderate change

- The top-3 set is unchanged for 4 of 8 genera (Bacteroides, Clostridium, Eubacterium, Prevotella).
- It changes for the other 4:
  - Blautia: choline and S-adenosyl-L-homocysteine are replaced by uridine and adenosine.
  - Collinsella: N-benzoylglycine is replaced by adenosine.
  - Eggerthella: arginine is replaced by 1,2-didecanoyl-sn-glycero-3-phosphocholine, which becomes its top compound.
  - Faecalibacterium: adenosine and uridine are replaced by 3-ureidopropanoate and dAMP.
- The SI summary statements partly change. Faecalibacterium is no longer a "nucleoside specialist", while MTA/SAH remain the commonest top compounds.
- **Supplementary Fig. 4c,** on the same 77 isolate–compound pairs: accuracy goes from 76.6% to 75.3% (28/3/30/16), and r from 0.555 to 0.48.
  - Choline is no longer in any genus's top 3. Without it (66 pairs), accuracy is 75.8% and r = 0.44.
  - The two new compounds (the phosphatidylcholine and dAMP) have no Venturelli measurements in the workbook, so they cannot be evaluated from the data we have.
- **Supplementary Fig. 4d:** by compound, choline goes from 91% to 73%, MTA from 91% to 82%, uridine from 36% to 45%, and arginine from 82% to 91%. By isolate, Blautia goes from 43% to 57%, Collinsella from 57% to 43%, and Eggerthella from 86% to 71%.

## Prompt 3 (Acinetobacter, NLDM): large change in the recommendations; consistent with Fig. S5

Ranks are among the 63 NLDM compounds that the agent ranked.

- **High score:** malate was 1st (13.06) and is now 8th (8.16).
  - Succinate rises from 4th to 1st (10.91), followed by uracil, glutamate, adenine and glycine.
  - Pyruvate falls from 6th to 15th (6.95).
  - Uracil is newly 2nd, and its limited solubility at 50 mM may lead the agent to set it aside, as it did adenine and guanine.
- **Neutral (score < 5, preferred by most microbes):** D-glucose still qualifies (2.39, 36th).
  - The transcript's example genera change: Streptomyces drops below 6 (from 6.42 to 4.95).
- **Lowest-score control:** spermidine is no longer the lowest (−1.46, 58th). L-ectoine is now the lowest (−3.98), with N-acetyl-L-lysine at −3.80. Both were the transcript's suggested alternatives.
- Fig. S5 already uses succinate, so a rerun of Prompt 3 should agree with the experiment better than the current transcript does.

Files in this directory: `prompt1_pseudomonas_ranking.csv`, `prompt2_top3.csv`, `prompt2_s4c_pairs.csv`,
`prompt2_s4d_by_compound.csv`, `prompt2_s4d_by_isolate.csv`, `prompt3_acinetobacter_nldm.csv`, `smoke_test_summary.json`.
