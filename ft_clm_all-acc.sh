#!/bin/bash

set -e

# mnli
bash ft_clm_mnli.sh > clm_mnli.out 2>&1 &&
# qnli
bash ft_clm_qnli.sh > clm_qnli.out 2>&1 &&
# rte
bash ft_clm_rte.sh > clm_rte.out 2>&1 &&
# sst2
bash ft_clm_sst2.sh > clm_sst2.out 2>&1 &&
# wnli
bash ft_clm_wnli.sh > clm_wnli.out 2>&1