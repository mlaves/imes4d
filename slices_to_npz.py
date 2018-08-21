from imes4d.utils import slices_to_npz
from tqdm import tqdm


if __name__ == "__main__":
    with tqdm(total=7) as pbar:
        slices_to_npz('/home/laves/Pictures/oct/Kugelplatte/0/*.JPG', 'sb_0.npz')
        pbar.update()
        slices_to_npz('/home/laves/Pictures/oct/Kugelplatte/1/*.JPG', 'sb_1.npz')
        pbar.update()
        slices_to_npz('/home/laves/Pictures/oct/Kugelplatte/2/*.JPG', 'sb_2.npz')
        pbar.update()
        slices_to_npz('/home/laves/Pictures/oct/Kugelplatte/3/*.JPG', 'sb_3.npz')
        pbar.update()
        slices_to_npz('/home/laves/Pictures/oct/Kugelplatte/4/*.JPG', 'sb_4.npz')
        pbar.update()
        slices_to_npz('/home/laves/Pictures/oct/Kugelplatte/5/*.JPG', 'sb_5.npz')
        pbar.update()
        slices_to_npz('/home/laves/Pictures/oct/Kugelplatte/6/*.JPG', 'sb_6.npz')
        pbar.update()
