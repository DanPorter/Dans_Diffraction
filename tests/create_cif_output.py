"""
Create file with string output of all included CIF files
"""

import os
import json

import Dans_Diffraction as dif

data_file = os.path.join(os.path.dirname(__file__), 'data', 'parsed_cif_output.json')

data = {xtl.name: str(xtl) for xtl in dif.structure_list}

json.dump(data, open(data_file, 'w'), indent=2)


for output in data.values():
    print(f"\n\n{output}")
