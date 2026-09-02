# Adding other platform policies
- To add other platform policies, create a new folder inside `policies/` folder.
- In that folder, provide the core prompt that includes community guidelines or rationale (if any) and the labeling policy details. Maintain the `Sx` label ID naming convention for uniformity. Additionally, add the `S0: no-moderation` class in the policy.
- Provide appropriate `output.md` file. Note the Bluesky output file contains mention fo `S10: other` which will not transfer to other platform policies. So, create a minor revised version and point to that. 
    - Additionally, the output file can be modified to prompt other things from the model. For instance, if some platform policy also allows for `action labels`, these can potentially be added to the prompt.