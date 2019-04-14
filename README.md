# 5526

qsub.sh is a simple script that removes the previous error and output logs, then starts another job.

job is a simple PBS job script. The first few lines are PBS directives (read more [on the OSC page here](https://www.osc.edu/supercomputing/batch-processing-at-osc/job-scripts) or run the command `man qsub` in the shell). The last few lines are the actual job to run. 

### The directives:
`-A` specifies the account. It's good practice to have and we should have the same one for this project.   
`-l` specifies what resources you need, including nodes, processors per node, gpus, and wall time (max time your program can run). Right now it's low so I can debug this setup quickly, but I plan on increasing it when I'm running the actual experiments.   
`-N` specifies the job name.   
`-e` (optional) specifies the error log file.  
`-o` (optional) specifies the output log file. You don't need these last two options but I the default names for the error/output logs are unwieldy. The reason you need these logs in the first place is because the job isn't run interactively (you could do that with the `-I` option. These makes the job more effecient and faster to schedule.  

### The code:
`cd $PBS_O_WORKDIR` moves to the directory where you submitted the job from. If you didn't include this, the job would start in your home directory. I start the job from inside the `rnn/` directory, so all my paths are based on that. 
`module load python` loads python in the job*.
`python trainer/imdbTask.py` runs my actual python code.
