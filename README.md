### Counterfactual Evaluation With SimBA

To train MACAW, run notebooks 02 and 03. You can then create and export counterfactuals using 06, and use 07 for calculating MSEs. 05 became a mess of different visualizations and tests, you can safely ignore it if you want.

To use the SFCN, either as a baseline or on counterfactuals, simply run the train file then the eval file, making sure to change any file locations or settings as needed.

You can use extract.py in /utils/ if you need to convert a new group's 3D images to 2D.

#### Bias subset folders
- far_bias: morphology bias in right hemisphere
- int_bias: moin_bias without morphology bias
- moin_bias: both morphology bias and intensity bias
- mor_bias: moin_bias without intensity bias
- near_bias: morphology bias in left hemisphere near disease region
- no_bias: no bias corresponding to far/near bias
- no_bias_moin: no bias corresponding to mor/int/moin bias
