import mujoco
import time
import itertools
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True, linewidth=100)

xml = """
<mujoco>
    <worldbody>
        <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
        <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
    </worldbody>
</mujoco>
"""


def main():
    model = mujoco.MjModel.from_xml_string(xml)
    sphere = model.geom("green_sphere")
    # __PRINT_VAR_START
    print(f"┆main┆ ╎sphere╎ ┊1┊: {str(sphere)}")  # __PRINT_VAR_END


if __name__ == "__main__":
    main()
