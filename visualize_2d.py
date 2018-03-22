import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
from scipy.ndimage import median_filter
from scipy.ndimage.interpolation import zoom


rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('font', **{'family': 'serif', 'serif': ['Times']})
rc('text', usetex=True)


if __name__ == '__main__':
    scale = 1
    a = np.load('stitched_total.npz')
    a = a[a.files[0]][::scale, ::scale, ::scale].astype(np.float32)
    a = np.transpose(a, (2, 1, 0))
    a = zoom(a, 2.0)

    #a = median_filter(a, size=5)
    fig = plt.figure(figsize=(a.shape[0]/320, a.shape[1]/320))

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
            print(i)
            fig.canvas.draw()

        return on_scroll


    cid = fig.canvas.mpl_connect('scroll_event', on_scroll_outer(0))
    plt.imshow(a[0])
    plt.show()

    fig.savefig('sphere_board_stitched_oct.pdf', bbox_inches='tight')
