# ANALYSIS AND OUTPUT INSTRUCTIONS

## Analysis
Carefully examine all post content against the community guidelines and labeling policy details to determine if the post does not need moderation (S0) or requires labeling for moderation (using labels except S0 from the Labeling Policy Details). Even if some content types are allowed on the platform, they might need to be moderated and appropriately labeled for safety and appropriate viewing.

## Output Fields

**PREDICTED_CATEGORY_ID**: Identify the single best matching label for moderation. Note:
 - Multiple labels are not allowed. If choosing no moderation needed (S0), no other moderation label should be chosen. 
 - Only use S10 as a last resort when the content needs moderation but does not fall into any other policy label. If some other moderation label generally fits the content, prefer that over S10.

**CONFIDENCE_SCORE**: Rate your confidence in the prediction on a 1–5 Likert scale:
  - 1: Not at all confident
  - 2: Somewhat low confidence
  - 3: Medium confidence
  - 4: Somewhat high confidence
  - 5: Very high confidence

## Required Format
Return your decision as a JSON object with exactly the following fields. Do not include any additional text as output outside the JSON.

{{
    "PREDICTED_CATEGORY_ID": "S0_if_no_moderation_or_single_label_S1_to_S10",
    "CONFIDENCE_SCORE": likert_scale_1_to_5
}}