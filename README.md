# 5526

qsub.sh is a simple script that removes the previous error and output logs, then starts another job.

job is a simple PBS job script. The first few lines are PBS directives (read more [on the OSC page here](https://www.osc.edu/supercomputing/batch-processing-at-osc/job-scripts) or run the command `man qsub` in the shell). The last few lines are the actual job to run. 

### The directives:
`-A` specifies the account. It's good practice to have and we should have the same one for this project.   
`-l` specifies what resources you need, including nodes, processors per node, gpus, and wall time (max time your program can run). Right now it's low so I can debug this setup quickly, but I plan on increasing it when I'm running the actual experiments.   
`-N` specifies the job name.   
`-e` (optional) specifies the error log file.  
`-o` (optional) specifies the output log file. You don't need these last two options but I the default names for the error/output logs are unwieldy. The reason you need these logs in the first place is because the job isn't run interactively (you could do that with the `-I` option. These makes the job more efficient and faster to schedule.  

### The code:
`cd $PBS_O_WORKDIR` moves to the directory where you submitted the job from. If you didn't include this, the job would start in your home directory. I start the job from inside the `rnn/` directory, so all my paths are based on that.   
`module load python` loads python in the job*. You could also specify here what version of python you want to load based on the modules available to you.   
`python trainer/imdbTask.py` runs my actual python code.   

*If you plan on using any special packages, like `tensorflow-gpu`, you'll need to install them to the default python module beforehand. This was easy for me, but if it doesn't work for you, you'll have to do some reading on the OSC website to figure it out.    
1. In the shell (not in a job), type `module avail python`. This will show you the different versions of python available to you. Assuming you're going to be using Python 3, then you want to see `python/3.6-conda5.2 (D)` somewhere in the list. `(D)` means that version of python is default, and will be the one that gets loaded when you enter `module load python`. If it doesn't, then you can type `module load python/3.6-conda5.2` to load it. 
2. Load the python module you want Assuming you loaded one with conda (a python package manager for scientific computing), use `conda install tensorflow-gpu` to install tensorflow-gpu. 
3. Verify installation by launching Python's shell and entering `import tensorflow`. You can check if a gpu is available by running an interactive job with a gpu and checking what devices tensorflow has available in the python shell.
