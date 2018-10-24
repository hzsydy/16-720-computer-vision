import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches

from SubtractDominantMotion import SubtractDominantMotion


# write your script here, we recommend the above libraries for making your animation

def main():
    data = np.load('../data/aerialseq.npy')
    h, w, n = data.shape
    # the animation
    fig, ax = plt.subplots(1, 1)
    ax.axis('off')

    h_im = ax.imshow(np.zeros((h, w, 3)), cmap='gray')

    cache = np.zeros((n, h, w, 3))

    def init():
        cache[0] = np.stack([data[:, :, 0]] * 3, axis=2)
        h_im.set_array(cache[0])
        return [h_im]

    def animate(i):
        print('frame', i)
        mask = SubtractDominantMotion(data[:, :, i], data[:, :, i + 1])
        cache[i + 1] = np.stack([data[:, :, i + 1]] * 3, axis=2)
        y, x = np.where(mask)
        cache[i + 1, y, x, :] = np.array([0, 0, 1])
        h_im.set_array(cache[i + 1])

        return [h_im]

    # call the animator.  blit=True means only re-draw the parts that have changed.
    ani = animation.FuncAnimation(fig, animate, init_func=init,
                                  frames=n - 1, interval=50, blit=True, repeat=False)
    plt.show()

    fig, axes = plt.subplots(1, 4)
    for i, c in enumerate([30, 60, 90, 120]):
        axes[i].axis('off')
        axes[i].imshow(cache[c], cmap='gray')
    plt.show()


if __name__ == '__main__':
    main()
