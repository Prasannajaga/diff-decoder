what is ARLM, MDLM & BD3LM ?

AR - autoregressive language model
MDLM -  Masked diffusion model
BD3LM - Block Discrete Denoising Diffusion Language model

assume the input: the cat sat on the table

## AR - Autorgressive Language model

The AR default to transformer casual attention
which predicts next token based on previous tokens

and it looks like

1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1

Token 1 (The) → sees nothing
Token 2 (cat) → sees The
Token 3 (sat) → sees The cat
Token 4 (on)  → sees The cat sat
Token 5 (the) → sees The cat sat on
Token 6 (table) → sees The cat sat on the

so the tokens only has the acess to the previous tokens
so it generate sequentially

token 1 -> token 2 -> .... token L

p(x)=i=1∏L​p(xi​∣x<i​)

## MDLM - Masked Diffusion model

the MDLM BERT style model uses bidirectional attention
which gives the access to see all tokens in the inputs and prediction happens parrelly

we usually add noise to the inputs before training:

t=0.5 t represents [0,1]

0 -> no noise
0.5 -> 50%
1 -> 100%

assume the input: the [mask] sat [mask] the [mask]
