# GenAlign
Generative Alignment for Multimodal Recommender

# baby 
nohup python main.py -m GenAlignGUME -d baby > ./output_GenAlignGUME_baby_20260127_exp1.log 2>&1 &

# sports
nohup python main.py -m GenAlignGUME -d sports > ../output_GenAlignGUME_sports_20260127_exp1.log 2>&1 &

# clothing
nohup python main.py -m GenAlignGUME -d clothing > ./output_GenAlignGUME_clothing_20260127_exp1.log 2>&1 &

# microlens
nohup python main.py -m GenAlignGUME -d microlens > ./output_GenAlignGUME_microlens_20260127_exp1.log 2>&1 &
