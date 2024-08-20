# easy demo
Step1 create environment using the following command: 

  conda env create -f environment.yml -n aimed
  
Step2 run demo.iypnb



# aimed
To Run the code you need to clone LDC repository and place the folder in the same folder of the trainer.py (This can be automatically done by executing the first cell of demo.ipynb).

Training data is also need to be placed in the niidata_c folder.

# Run
You can train your own model with:

python trainer.py --device_number 0

evaluator is under construction
