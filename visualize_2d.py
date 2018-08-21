import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
from scipy.ndimage import median_filter
from scipy.ndimage.interpolation import zoom


params = {'text.usetex': True,
          'text.latex.unicode': True,
          'font.size': 10,
          'font.serif': 'Times',
          'font.sans-serif': 'Helvetica',
          }
plt.rcParams.update(params)


if __name__ == '__main__':
    scale = 1
    a = np.load('data/stitched_total_sphereboard.npz')
    a = a[a.files[0]][::scale, ::scale, ::scale].astype(np.float32)
    a = np.transpose(a, (2, 1, 0))  # use this for sphere board dataset
    a = zoom(a, 2.0)

    #a = median_filter(a, size=5)
    fig = plt.figure(figsize=(a.shape[0]/290, a.shape[1]/290))

    def on_scroll_outer(i):

        def on_scroll(event):
            nonlocal i
            if 'up' in event.button:
                i = i + 1
            elif 'down' in event.button:
                i = i - 1
            if i < 0:
                i = 0
            elif i > a.shape[0]-1:
                i = a.shape[0]-1
            plt.clf()
            plt.imshow(a[i])

            # plot volume bounds
            plt.axvline(x=2 * 256, linewidth=0.5, color=(0.8, 0.4, 0.4), linestyle='--')

            plt.axvline(x=2 * 68, linewidth=0.5, color=(0.871, 0.576, 0.373), linestyle='--')
            plt.axvline(x=2 * 320, linewidth=0.5, color=(0.871, 0.576, 0.373), linestyle='--')

            plt.axvline(x=2 * 134, linewidth=0.5, color=(0.71, 0.741, 0.408), linestyle='--')
            plt.axvline(x=2 * 387, linewidth=0.5, color=(0.71, 0.741, 0.408), linestyle='--')

            plt.axvline(x=2 * 199, linewidth=0.5, color=(0.157, 0.165, 0.18), linestyle='--')
            plt.axvline(x=2 * 453, linewidth=0.5, color=(0.157, 0.165, 0.18), linestyle='--')

            plt.axvline(x=2 * 265, linewidth=0.5, color=(0.541, 0.745, 0.718), linestyle='--')
            plt.axvline(x=2 * 519, linewidth=0.5, color=(0.541, 0.745, 0.718), linestyle='--')

            plt.axvline(x=2 * 330, linewidth=0.5, color=(0.506, 0.635, 0.745), linestyle='--')
            plt.axvline(x=2 * 585, linewidth=0.5, color=(0.506, 0.635, 0.745), linestyle='--')

            plt.axvline(x=2 * 389, linewidth=0.5, color=(0.698, 0.58, 0.733), linestyle='--')

            print(i)
            fig.canvas.draw()

        return on_scroll


    cid = fig.canvas.mpl_connect('scroll_event', on_scroll_outer(0))
    plt.imshow(a[0])
    plt.show()

    fig.savefig('sphere_board_stitched_oct.pdf', bbox_inches='tight')
