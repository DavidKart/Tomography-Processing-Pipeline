import os
import sys
import shutil
import re
import json

def predict(cryoCAREpath, mapsList):
    
    os.chdir(cryoCAREpath)
    rawTomos = "rawTomos"
    if not os.path.isdir("../other/cryoCare_blueprint"):
        print("Directory ../other/cryoCare_blueprint does not exist. Wrong Setup. Exiting.")
        sys.exit(1)
        
    cryo_care_blueprint = os.path.realpath("../other/cryoCare_blueprint")
    os.chdir(rawTomos)
    
    name = "runForAll"
    new_directory_path = os.path.join("..", name)

    # Copy
    source = os.path.join(cryo_care_blueprint, "predict_config.json")
    destination = os.path.join(new_directory_path, "predict_config.json")
    if not os.path.exists(destination): 
        shutil.copy2(source, destination)
            
    even_halfs = []
    odd_halfs = []

    for x in mapsList:
        matching_files = [f for f in os.listdir(".") if re.match(rf"even_.*{re.escape(x)}_rec_tomo\.mrc", f)]
        if not matching_files:
            print(f"{x} does not have associated half maps. Probably, something during preprocessing went wrong (AreTomo failed).")
        else:
            i = matching_files[0]
            
            odd_half = re.sub(r'even', 'odd', i, count=1)

            realpath_even = os.path.realpath(i)
            realpath_odd = os.path.realpath(odd_half)
            
            even_halfs.append(f'{realpath_even}')
            odd_halfs.append(f'{realpath_odd}')

    #escape double quotes where necessary
    for ind_i in range(len(even_halfs)):
        even_halfs[ind_i] = even_halfs[ind_i].replace('"', '\\"')
  
    for ind_i in range(len(odd_halfs)):
        odd_halfs[ind_i] = odd_halfs[ind_i].replace('"', '\\"')
                
    os.chdir(f"../{name}")   
     
    with open("predict_config.json", 'r') as f:
        data = json.load(f)

        data["even"] = even_halfs
        data["odd"] = odd_halfs
 
    with open("predict_config.json", 'w') as f:
        json.dump(data, f, indent=4)
        
    os.system("cryoCARE_predict.py --conf predict_config.json")
