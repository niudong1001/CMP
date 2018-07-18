import deepmind_lab
import numpy as np

# Create a new environment object.
lab = deepmind_lab.Lab("tests/empty_room_test", ['RGB_INTERLEAVED'],
                       {'fps': '30', 'width': '80', 'height': '60'})
lab.reset(seed=1)
obs= lab.observations()
# print(obs)
# Execute 100 walk-forward steps and sum the returned rewards from each step.
print sum(
    [lab.step(np.array([0,0,0,1,0,0,0], dtype=np.intc)) for i in range(0, 100)])