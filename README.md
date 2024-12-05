# ViTproject

 this repository is supported by Arivazhagan G. Balasubramanian and this is only reproduction of his work.
For further information, check https://arxiv.org/abs/2404.14121
In order to reproduce this work easily, you ought to execute the following commands.
## setting 
First, create virtual environment to execute the work.PLEASE NOTE that before you execute python -m venv <env name>, load the module first.

```
module load TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1
python -m venv vit
```
"vit" the name of the virtual environment. If you want, you can replace the name and change the batch script.  
Second, activate the virtual environment and install all the necessary libraries at once. do
```
source vit bin activate
pip install -r requirements.txt
```
initial setting is over! after that you can train&evaluate the models.
## Training
you can creating the model by 
```
sbatch train.sh
```
you can check the training state by 
```
gbck
```
if the USER ST shows "PD", do
```
check <your username>
```
the models are saved in .saved_models folder, and the checkpoints are preserved in .logs folder.In .epoch_log, trainging information was preserved and by
```
python plot.py
``` 
you can get epoch.pdf, that shows the training curve of models.

## Evaluation
in order to evaluate the specific models, you have to specify which model to evaluate.
PLEASE REMEMBER the current time you execute train.sh in CTC time and specify. This information must be written in order to evaluate. Such information should be written in config_deep.py
after you edit, do
```
sbatch eval.sh
```
if the script works well, the directory .predictions will be generated and inside of it 
```
.predictions/NN_WakkR../pled_fluct0000.npz
```
will be generated.
inside of it, the prediction result exists.
in order to get the performance of it in the forms of .txt file, you specify such .npz file in the NN_velocity_result.py and do
```
sbatch postprocess.sh
```
and you will find 
```
tmp/result.txt
```
change the hyperparameters and check what will happen. 
After you understand the structure, let's try editing the models. 

this is the machine-learning training code and this code is originally written in tensorflow. I wrote in pytorch but the mechanics is same; both library uses the architechture of Convolutional Neural Networks(CNN).  
And, this process is done via super GPU computer. to request some program, execute  
```
sbatch submit.sh
```
what is sbatch? -see [link](https://slurm.schedmd.com/sbatch.html)
```
squeue -u geethaa
```
or
```
gbck
```
shows the job progress
lsr

show the training situation -> look at the train/train.txt
## note
pytorch is not available. Learn how to use Tensorflow.
how can we evaluate the model? -> CNN-predict.py, which generates pred_<number>.npz files.
and result.txt can be generated via NN_velocity_result.py, which generates result.txt.
and all the scripts are executed on .sh files. Check 
```
train.sh, eval.sh, postprocess.sh
```

submit.sh is not the suitable word for it: I think training.sh and eval.sh should be prepared so that anyone can work and measure the efficiency of the model. 
```
projinfo 
```
means other members using priority
shows 
```
scancel 2799829
```
means cancelling request
check the results : 
```
cd .jupyter_plots
```
if there is something lost packages, execute
```
module spider <module name>
```
you can search the module name by 
```
```
scontrol is 
see here if you want to make a new virtual environment https://www.c3se.chalmers.se/documentation/applications/python/#virtual-environments
I tried some techniques of Normalization, such as Batch normalization and Layer Normalization. However, when it comes to layer normalization, the performance is decreased dramatically. conv2dcompose and layer normalization should not be mixed.
# warning!
## the coordinates are a little bit strange. 
after you processed 
```
sbatch postprocess.sh
```
you will get tmp/result.txt and read the outcome. However, you should be careful that the coordinates are permutated. 
the file says "v ,w, u" ,which is wrong. you HAVE to read them as "u, v, w" (x,y,z)
Please ignore 
## ml models
*.h5
*.pt
*.ckpt

## other large files
*.zip
*.tar.gz

# explanations of functions
input_parser: This is a parser function. It defines the template for
    interpreting the examples you're feeding in. Basically, 
    this function defines what the labels and data look like
    for your labeled data. 
output_parser:This is a parser function. It defines the template for
    interpreting the examples you're feeding in. Basically, 
    this function defines what the labels and data look like
    for your labeled data. 

# Versions
used keras new version and it is now impossible to use 
```
decay = self.model.optimizer.decay
```


