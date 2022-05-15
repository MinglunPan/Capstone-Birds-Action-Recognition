# MSCA-Capstone-Birds-Action-Recognition

This file will be updated in the future.


## Brief Introduction

Step 1. use `sbatch msca-gpu.batch` to request the resources

Step 2. Use `squeue`, `sinfo` to see if the job has submitted successfully.

Step 3. If the job has been allocated the resources requested, you will see a file named `notebook.log` on the same folder. And use cat notebook.log, you could get the url of jupyter notebook.

REMEMBER: When you DO NOT need the resources any more, use `scancel <jobid>` to release the resources occupied. `<jobid>` can be seen by `squeue -u <CNET_ID>`

  
