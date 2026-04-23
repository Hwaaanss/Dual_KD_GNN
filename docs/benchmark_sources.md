# Benchmark Source Notes

This document records the primary source used while modularizing the notebook models.

## Alignment summary

- `attentive_fp`: switched to the official `torch_geometric.nn.models.AttentiveFP` implementation so the modularized code follows the published AttentiveFP architecture directly.
- `d_mpnn`: kept as a lightweight local implementation of directed bond message passing with descriptor concatenation, following the D-MPNN / Chemprop paper structure without vendoring the full Chemprop training stack.
- `fp_gnn`: kept as a local modular implementation that preserves the official FP-GNN paper and repository pattern of combining a graph branch with a fingerprint branch.
- `ml_mpnn`: kept as a paper-guided local implementation of the multi-level message passing architecture from AdvProp.
- `mlfgnn`: kept as a paper-guided local implementation of the multi-level fusion architecture; no official GitHub repository was identified during this refactor.
- `multichem`: kept as a local modular implementation guided by the official MultiChem paper and repository layout.
- `dual_kd_gnn`: original experimental architecture from the notebook, modularized without changing its core design.

## Primary sources

- AttentiveFP paper: https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959
- PyG AttentiveFP reference: https://github.com/pyg-team/pytorch_geometric
- D-MPNN / Chemprop paper: https://chemrxiv.org/engage/chemrxiv/article-details/60c743f4bb8c1a1f7d3da3f9
- Chemprop repository: https://github.com/chemprop/chemprop
- FP-GNN paper: https://academic.oup.com/bib/article/23/6/bbac408/6702671
- FP-GNN repository: https://github.com/idrugLab/FP-GNN
- ML-MPNN paper within AdvProp: https://academic.oup.com/bioinformatics/article-abstract/38/9/2579/6531963
- MLFGNN paper: https://pubs.acs.org/doi/abs/10.1021/acs.jcim.5c01525
- MultiChem paper: https://biodatamining.biomedcentral.com/articles/10.1186/s13040-024-00419-4
- MultiChem repository: https://github.com/DMnBI/MultiChem
