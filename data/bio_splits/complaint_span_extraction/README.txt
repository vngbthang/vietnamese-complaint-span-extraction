Vietnamese Complaint Span Extraction Dataset
============================================

Status: NOT YET ANNOTATED

ViOCD contains review-level complaint labels (0/1) but NO complaint span
annotations. To create a complaint_span_extraction dataset:

1. Annotate complaint spans (character-level [start, end]) in ViOCD texts.
2. Validate span offsets (start >= 0, end <= len(text), text[start:end] matches).
3. Run BIO conversion with labels O, B-COMP, I-COMP.
4. Perform round-trip validation.

Until step 3 is complete, complaint_span_extraction has no BIO data.
