#!/bin/bash

rsync -avh -e "ssh -i ~/.ssh/gbar"  s173981@transfer.gbar.dtu.dk:~/Phonon_cQED/Polariton_Laser/Corr/ Corr/

