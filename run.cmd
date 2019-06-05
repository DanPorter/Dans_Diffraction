:: Start an interactive terminal session using Dans_Diffraction


ipython -i --matplotlib=tk -c "import numpy as np; import matplotlib.pyplot as plt; import Dans_Diffraction as dif; xtl=dif.structure_list.Ca2RuO4.build()"


echo "Finished"