import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches

from LucasKanadeBasis import LucasKanadeBasis
from LucasKanade import LucasKanade

def main():
    data = np.load('../data/sylvseq.npy')
    base = np.load('../data/sylvbases.npy')
    rect = np.array([101, 61, 155+1, 107+1], dtype=np.float)
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
    h_rect_sb = ax.add_patch(patches.Rectangle((0, 0), 0, 0, fill=False, color='g'))
    h_template = ax_template.imshow(data[61:107+1, 101:155+1, 0], cmap='gray')

    def init():
        h_im.set_array(data[:, :, 0])
        p = (rect[0], rect[1])
        h_rect.set_xy(p)
        h_rect.set_height(hh)
        h_rect.set_width(ww)
        h_rect_sb.set_xy(p)
        h_rect_sb.set_height(hh)
        h_rect_sb.set_width(ww)
        return [h_im, h_rect, h_template, h_rect_sb]

    rect_sb = rect.copy()
    rects_sb = np.zeros((n, 4))
    rects_sb[0] = rect_sb.copy()
    template_idx = 0
    def animate(i):
        nonlocal rect
        nonlocal template_idx
        print('frame', i)
        template = data[int(rect[1] + 0.5):int(rect[1] + 0.5 + hh + 1e-5),
                   int(rect[0] + 0.5):int(rect[0] + 0.5 + hh + 1e-5), i + 1]
        h_template.set_array(template)
        h_im.set_array(data[:, :, i])

        p = LucasKanadeBasis(data[:, :, i], data[:, :, i + 1], rect, base)
        rect[0::2] += p[0]
        rect[1::2] += p[1]
        rects[i] = rect.copy()
        h_rect.set_xy((rect[0], rect[1]))

        p = rects_sb[i, :2] - rects_sb[template_idx, :2]
        pn = LucasKanade(data[:, :, template_idx], data[:, :, i + 1], rects_sb[template_idx], p)
        if i > 0:
            pn += rects_sb[template_idx, :2] - rects_sb[0, :2]
            pn_star = LucasKanade(data[:, :, 0], data[:, :, i + 1], rects_sb[0], pn)
            if np.linalg.norm(pn_star - pn) < 1e-3:
                template_idx = i + 1
            else:
                template_idx = i
            pn = pn_star
        else:
            template_idx = 1

        rects_sb[i + 1][0::2] = pn[0]
        rects_sb[i + 1][1::2] = pn[1]
        rects_sb[i + 1] += rects_sb[0]
        h_rect_sb.set_xy((rects_sb[i + 1, 0], rects_sb[i + 1, 1]))

        return [h_im, h_rect, h_template, h_rect_sb]

    # call the animator.  blit=True means only re-draw the parts that have changed.
    ani = animation.FuncAnimation(fig, animate, init_func=init,
                                  frames=n - 1, interval=50, blit=True, repeat=False)
    plt.show()

    fig, axes = plt.subplots(1, 5)
    for i, c in enumerate([1, 200, 300, 350, 400]):
        axes[i].axis('off')
        axes[i].imshow(data[:, :, c], cmap='gray')
        x1, y1, x2, y2 = rects[c]
        axes[i].add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='y'))
        x1, y1, x2, y2 = rects_sb[c]
        axes[i].add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='g'))
    plt.show()

    np.save('sylvseqrects.npy', rects)


if __name__ == '__main__':
    main()
