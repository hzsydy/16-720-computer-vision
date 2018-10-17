import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from LucasKanade import LucasKanade, interpTemplate

def main():
    data = np.load('../data/carseq.npy')
    rect = np.array([59, 116, 145, 151], dtype=np.float)
    tol = 0.1
    h, w, n = data.shape

    fig, (ax, ax_last, ax_template) = plt.subplots(1, 3)
    ax.axis('off')
    ax_last.axis('off')
    ax_template.axis('off')

    h_im = ax.imshow(data[:, :, 0], cmap='gray')
    h_rect = ax.add_patch(patches.Rectangle((0,0), 0, 0, fill=False))
    h_im_last = ax_last.imshow(data[:, :, 0], cmap='gray')
    h_rect_last = ax_last.add_patch(patches.Rectangle((0,0), 0, 0, fill=False))
    h_template = ax_template.imshow(data[116:151, 59:145, 0], cmap='gray')

    ww, hh = rect[2] - rect[0], rect[3] - rect[1]

    def init():
        h_im.set_array(data[:, :, 0])
        h_im_last.set_array(data[:, :, 0])
        p = (rect[0], rect[1])
        h_rect.set_xy(p)
        h_rect.set_height(hh)
        h_rect.set_width(ww)
        h_rect_last.set_xy(p)
        h_rect_last.set_height(hh)
        h_rect_last.set_width(ww)
        return [h_im, h_im_last, h_rect, h_rect_last, h_template]


    p = np.zeros(2)

    def animate(i):
        nonlocal p
        print('frame', i)
        h_im_last.set_array(data[:, :, i])
        h_rect_last.set_xy((rect[0], rect[1]))
        template = data[int(rect[1]+0.5):int(rect[1]+0.5+hh+1e-5), int(rect[0]+0.5):int(rect[0]+0.5+hh+1e-5), i+1]
        h_template.set_array(template)
        h_im.set_array(data[:, :, i])
        rect[0::2] += p[0]
        rect[1::2] += p[1]
        h_rect.set_xy((rect[0], rect[1]))
        return [h_im, h_im_last, h_rect, h_rect_last, h_template]


    # call the animator.  blit=True means only re-draw the parts that have changed.
    ani = animation.FuncAnimation(fig, animate, init_func=init,
                                  frames=n-1, interval=50, blit=True)
    plt.show()

if __name__ == '__main__':
    main()
