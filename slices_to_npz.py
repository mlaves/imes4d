from imes4d.utils import slices_to_npz
from tqdm import tqdm


if __name__ == "__main__":
    with tqdm(total=4) as pbar:
        slices_to_npz('/home/laves/Pictures/oct/Felsenbein/0/*.JPG', '0.npz')
        pbar.update()
        slices_to_npz('/home/laves/Pictures/oct/Felsenbein/1/*.JPG', '1.npz')
        pbar.update()
        slices_to_npz('/home/laves/Pictures/oct/Felsenbein/2/*.JPG', '2.npz')
        pbar.update()
        slices_to_npz('/home/laves/Pictures/oct/Felsenbein/3/*.JPG', '3.npz')
        pbar.update()
        # slices_to_npz('/home/laves/Pictures/oct/Kugelplatte/4/*.JPG', '4.npz')
        # pbar.update()
        # slices_to_npz('/home/laves/Pictures/oct/Kugelplatte/5/*.JPG', '5.npz')
        # pbar.update()
        # slices_to_npz('/home/laves/Pictures/oct/Kugelplatte/6/*.JPG', '6.npz')
        # pbar.update()
        # slices_to_npz('/home/laves/Pictures/oct/Kugelplatte/7/*.JPG', '7.npz')
        # pbar.update()
        # slices_to_npz('/home/laves/Pictures/oct/Kugelplatte/8/*.JPG', '8.npz')
        # pbar.update()
