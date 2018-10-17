import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from LucasKanade import LucasKanade


def main():
    data = np.load('../data/carseq.npy')
    rect = np.array([59, 116, 145+1, 151+1], dtype=np.float)
    tol = 0.1
    h, w, n = data.shape

    ww, hh = rect[2] - rect[0], rect[3] - rect[1]
    rects = np.zeros((n, 4))
    rects[0] = rect.copy()

    # the animation
    fig, (ax, ax_template) = plt.subplots(1, 2)
    ax.axis('off')
    ax_template.axis('off')

    h_im = ax.imshow(data[:, :, 0], cmap='gray')
    h_rect = ax.add_patch(patches.Rectangle((0, 0), 0, 0, fill=False, color='y'))
    h_template = ax_template.imshow(data[116:151+1, 59:145+1, 0], cmap='gray')
    def init():
        h_im.set_array(data[:, :, 0])
        p = (rect[0], rect[1])
        h_rect.set_xy(p)
        h_rect.set_height(hh)
        h_rect.set_width(ww)
        return [h_im, h_rect, h_template]

    def animate(i):
        print('frame', i)
        template = data[int(rect[1] + 0.5):int(rect[1] + 0.5 + hh + 1e-5),
                   int(rect[0] + 0.5):int(rect[0] + 0.5 + hh + 1e-5), i + 1]
        h_template.set_array(template)
        h_im.set_array(data[:, :, i])
        p = LucasKanade(data[:, :, i], data[:, :, i + 1], rect)
        rect[0::2] += p[0]
        rect[1::2] += p[1]
        rects[i+1] = rect.copy()
        h_rect.set_xy((rect[0], rect[1]))
        return [h_im, h_rect, h_template]

    # call the animator.  blit=True means only re-draw the parts that have changed.
    ani = animation.FuncAnimation(fig, animate, init_func=init,
                                  frames=n - 1, interval=50, blit=True, repeat=False)
    plt.show()

    fig, axes = plt.subplots(1, 5)
    for i in range(5):
        axes[i].axis('off')
        axes[i].imshow(data[:,:,i*100+1], cmap='gray')
        x1, y1, x2, y2 = rects[i*100+1]
        axes[i].add_patch(patches.Rectangle((x1,y1), x2-x1, y2-y1, fill=False, color='y'))
    plt.show()

    np.save('carseqrects.npy', rects)

if __name__ == '__main__':
    main()
