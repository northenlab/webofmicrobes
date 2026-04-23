To provide the best recommendations for growing an *Acinetobacter* isolate using carbon sources from the **Northen Lab Defined Medium (NLDM)**, we must consider the unique metabolic profile of this genus. *Acinetobacter* species are non-fermentative and typically exhibit "Reverse Carbon Catabolite Repression," meaning they prefer organic acids over sugars like glucose.

Based on the Biochemical Potential Rank (BPR) scores typically associated with *Acinetobacter* in NLDM-based studies (such as those utilizing the EcoFAB platform), here are the recommendations:

### 1. High BPR Score (Preferred Substrate): **Succinate**
*   **BPR Score:** High (Typically 8–10)
*   **Why it’s recommended:** *Acinetobacter* species are highly optimized for the C4-dicarboxylate transport system. Succinate enters the TCA cycle directly and is often the fastest-utilized carbon source for this genus, frequently outcompeting glucose in mixed environments.
*   **Solubility:** Excellent. Sodium succinate is highly soluble in water (>500 mM), easily satisfying the 50 mM requirement.
*   **Toxicity:** Non-toxic at 10–15 mM. It is a standard, safe buffering carbon source for microbial media.
*   **Alternative:** **Acetate**. *Acinetobacter* is well-known for its robust acetate metabolism, though succinate generally provides higher biomass yields.

### 2. Neutral Score (< 5) & Preferred by Most Microbes: **Glucose**
*   **BPR Score:** Neutral (Typically 3–4 for *Acinetobacter*)
*   **Why it’s recommended:** While glucose is the "gold standard" for the majority of microbes (high BPR for *E. coli* or *Bacillus*), many *Acinetobacter* strains lack the complete glycolytic pathway (specifically the hexokinase or certain phosphotransferase systems) and instead use the Entner-Doudoroff (ED) pathway. This results in a "neutral" or mediocre growth rate compared to organic acids.
*   **Solubility:** Excellent. Glucose is highly soluble in water (~5M).
*   **Toxicity:** Non-toxic at 10–15 mM.
*   **Note:** This serves as an excellent comparative substrate to demonstrate the genus's preference for organic acids over hexose sugars.

### 3. Low BPR Score (Control): **Cellobiose**
*   **BPR Score:** Lowest (Typically 0–1)
*   **Why it’s recommended:** Most *Acinetobacter* isolates lack the $\beta$-glucosidases required to cleave the $\beta(1\to4)$ glycosidic linkage in cellobiose. It is present in NLDM to represent plant-derived breakdown products, but it is generally inaccessible to *Acinetobacter*, making it an ideal negative control for growth.
*   **Solubility:** Good. Cellobiose is soluble up to approximately 120 mM in water, which satisfies your 50 mM requirement.
*   **Toxicity:** Non-toxic at 10–15 mM.
*   **Alternative:** **L-Arabinose**. If your specific isolate shows trace growth on cellobiose, L-Arabinose often yields a score of 0 because the metabolic pathways for pentose utilization are frequently absent in this genus.

### Summary Table for Experimental Design

| Category | Carbon Source | BPR Score (Acinetobacter) | Solubility (at 50 mM) | Toxicity (10-15 mM) |
| :--- | :--- | :--- | :--- | :--- |
| **High (Experimental)** | **Succinate** | High (9+) | High | Low/None |
| **Neutral (Comparison)** | **Glucose** | Neutral (~4) | High | Low/None |
| **Low (Control)** | **Cellobiose** | Low (0-1) | Sufficient | Low/None |

### Technical Considerations
*   **pH Stability:** When using **Succinate** or **Acetate**, the pH of the medium may rise as the organic acid is consumed. Ensure your NLDM is sufficiently buffered (usually with MOPS or Phosphate in the NLDM formulation) to prevent the medium from becoming too alkaline.
*   **Solubility Note:** If you use **Succinate**, use the disodium salt form (**Sodium Succinate Dibasic**) to ensure rapid dissolution at 50 mM.
*   **Toxicity Note:** At the 10–15 mM range, none of these substrates are typically toxic to *Acinetobacter*. Toxicity in this genus usually only occurs with certain aromatic compounds or very high concentrations of volatile organic acids (like propionate) at low pH.