import sys
import math
import h5py
from OCC.Core.AIS import AIS_Shape
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.gp import gp_Pnt
from cadlib.visualize import vec2CADsolid
from OCC.Core.BRepBndLib import brepbndlib_Add
from OCC.Display.SimpleGui import init_display


def visualize_and_save_with_rotations(h5file, image_root):
    try:
        with h5py.File(h5file, "r") as fp:
            out_vec = fp["vec"][:]  # (len, 1 + N_ARGS)

        compound = vec2CADsolid(out_vec)

        display, start_display, add_menu, add_function_to_menu = init_display()

        display.SetBackgroundImage("./data/white.png")

        ais_shape = AIS_Shape(compound)

        display.Context.Display(ais_shape, True)

        display.View.TriedronErase()

        bbox = Bnd_Box()
        brepbndlib_Add(compound, bbox)

        display.View_Iso()
        display.FitAll()

        display.View.SetProj(1, -1, 1)

        rotations = [
            (30, 0, 0),
            (0, 30, 0),
            (0, 0, 30),
            (30, 30, 0),
            (30, 0, 30),
            (0, 30, 30),
            (30, 30, 30),
        ]

        for idx, (x_rot, y_rot, z_rot) in enumerate(rotations):
            if x_rot != 0:
                display.View.Rotate(math.radians(x_rot), True)

            if y_rot != 0:
                display.View.Rotate(math.radians(y_rot), True)

            if z_rot != 0:
                display.View.Rotate(math.radians(z_rot), True)

            bbox = Bnd_Box()
            brepbndlib_Add(compound, bbox)
            xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()

            center_x = (xmin + xmax) / 2
            center_y = (ymin + ymax) / 2
            center_z = (zmin + zmax) / 2
            center_point = gp_Pnt(center_x, center_y, center_z)

            display.FitAll()

            output_image_file = f"{image_root}/rotate({x_rot},{y_rot},{z_rot}).png"

            display.View.Dump(output_image_file)
            print(f"�̹��� ���� �Ϸ�: {output_image_file}")

        display.EraseAll()  #

        del display

    except Exception as e:
        log_error(h5file)
        print(f"Error: {str(e)}")


def log_error(h5file, error_log_file="error_log.txt"):
    with open(error_log_file, "a") as log_file:
        log_file.write(f"Error processing file: {h5file}\n")
        print(f"Error logged for file: {h5file}")


if __name__ == "__main__":
    h5file = sys.argv[1]
    image_root = sys.argv[2]

    visualize_and_save_with_rotations(h5file, image_root)
