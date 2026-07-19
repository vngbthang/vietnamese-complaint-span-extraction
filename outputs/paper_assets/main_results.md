# Main Results — Vietnamese Complaint Span Extraction

| setting_type | method | auxiliary_data | entity_precision | entity_recall | entity_f1 | token_macro_f1 | token_accuracy | total_runtime | is_best |
|---|---|---|---|---|---|---|---|---|---|
| transfer | uvisd4sa_then_complaint | UIT-ViSD4SA | 0.5338 | 0.4225 | 0.4717 | 0.9137 | 0.8516 | 1256.2876 | **BEST** |
| transfer | aux_all_then_complaint | UIT-ViSD4SA + CausaSent-ATE-v2 | 0.5033 | 0.4282 | 0.4627 | 0.9129 | 0.8506 | 1712.6591 |  |
| direct | phobert_weighted_ce | none | 0.4899 | 0.4113 | 0.4472 | 0.9107 | 0.8471 | — |  |
| direct | mbert_ce | none | 0.4820 | 0.3775 | 0.4234 | 0.9004 | 0.8292 | — |  |
| transfer | causasent_then_complaint | CausaSent-ATE-v2 | 0.4560 | 0.3944 | 0.4230 | 0.9091 | 0.8447 | 1082.2784 |  |
| direct | xlm_roberta_ce | none | 0.5144 | 0.3521 | 0.4181 | 0.9096 | 0.8452 | — |  |
| direct | phobert_ce | none | 0.4797 | 0.3662 | 0.4153 | 0.9098 | 0.8450 | — |  |
