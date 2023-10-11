This code was run on Python version 3.10. To initialize your environment, please run 
```bash
bash setup.sh
```
if you have a GPU. Otherwise, please run 
```bash
bash setup_for_cpu.sh
```

Then, to recreate any of our experiments, please run
```bash
python main.py -c cfgs/[name of yaml] --seed [seed]
```
All of our experimental setups are saved in the cfgs folder. We detail all the folders and the experiments in each.
cfgs/best: all of our method's experiments
cfgs/cb_baseline: experiments for Conformal Bayes
cfgs/chr: experiments for CHR
cfgs/cqr: experiments for CQR
cfgs/mle: experiments for MLE ablation
cfgs/mle_entropy: experiments for MLE with Entropy ablation
cfgs/heteroskedastic: experiments to recreate the heteroskedastic figures 
cfgs/lei_baseline: experiments to recreate the KDE from Lei 2014
cfgs/no_entropy: experiments for ablation without entropy
cfgs/ridge_baseline: experiments for Lasso Method from Lei 2019

For details on how to get MEPs data, please refer to https://github.com/yromano/cqr/tree/master/get_meps_data
 