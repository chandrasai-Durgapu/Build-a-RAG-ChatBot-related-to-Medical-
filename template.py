import os 
from pathlib import Path



list_of_files = [

    f"src/__init__.py",
    f"src/helper.py",  
    f"src/prompt.py",
    ".env",
    
    f"research/trails.ipynb",
   
    "app.py",
    "requirements.txt",
    
    "setup.py",
   


]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir!= "":
        os.makedirs(filedir, exist_ok=True)
    
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):
        with open(filepath, 'w') as f:
            pass

    else:
        print(f"{filename} is already present in {filedir} and has some content. Skipping creation.")