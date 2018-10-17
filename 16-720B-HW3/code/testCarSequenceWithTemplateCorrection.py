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
    rects_sb = np.load('carseqrects.npy')
    rects[0] = rect.copy()

    # the animation
    fig, (ax, ax_template) = plt.subplots(1, 2)
    ax.axis('off')
    ax_template.axis('off')

    h_im = ax.imshow(data[:, :, 0], cmap='gray')
    h_rect = ax.add_patch(patches.Rectangle((0, 0), 0, 0, fill=False, color='y'))
    h_rect_sb = ax.add_patch(patches.Rectangle((0, 0), 0, 0, fill=False, color='g'))
    h_template = ax_template.imshow(data[116:151+1, 59:145+1, 0], cmap='gray')

    def init():
        h_im.set_array(data[:, :, 0])
        p = (rect[0], rect[1])
        h_rect.set_xy(p)
        h_rect.set_height(hh)
        h_rect.set_width(ww)
        h_rect_sb.set_xy(p)
        h_rect_sb.set_height(hh)
        h_rect_sb.set_width(ww)
        return [h_im, h_rect, h_rect_sb, h_template]

    template_idx = 0

    def animate(i):
        nonlocal template_idx
        print('frame', i)
        rect = rects[template_idx]
        template = data[int(rect[1] + 0.5):int(rect[1] + 0.5 + hh + 1e-5),
                   int(rect[0] + 0.5):int(rect[0] + 0.5 + hh + 1e-5), i + 1]
        h_template.set_array(template)
        h_im.set_array(data[:, :, i])
        p = rects[i, :2] - rects[template_idx, :2]
        pn = LucasKanade(data[:, :, template_idx], data[:, :, i + 1], rects[template_idx], p)
        if i > 0:
            pn += rects[template_idx, :2] - rects[0, :2]
            pn_star = LucasKanade(data[:, :, 0], data[:, :, i + 1], rects[0], pn)
            if np.linalg.norm(pn_star - pn) < 1e-3:
                template_idx = i + 1
            else:
                template_idx = i
        else:
            template_idx = 1

        rects[i + 1][0::2] = pn[0]
        rects[i + 1][1::2] = pn[1]
        rects[i + 1] += rects[0]
        h_rect.set_xy((rects[i + 1, 0], rects[i + 1, 1]))
        h_rect_sb.set_xy((rects_sb[i + 1, 0], rects_sb[i + 1, 1]))
        return [h_im, h_rect, h_rect_sb, h_template]

    # call the animator.  blit=True means only re-draw the parts that have changed.
    ani = animation.FuncAnimation(fig, animate, init_func=init,
                                  frames=n - 1, interval=50, blit=True, repeat=False)
    plt.show()

    fig, axes = plt.subplots(1, 5)
    for i, c in enumerate([1, 100, 200, 300, 400]):
        axes[i].axis('off')
        axes[i].imshow(data[:, :, c], cmap='gray')
        x1, y1, x2, y2 = rects[c]
        axes[i].add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='y'))
        x1, y1, x2, y2 = rects_sb[c]
        axes[i].add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='g'))
    plt.show()

    np.save('carseqrects-wcrt.npy', rects)


if __name__ == '__main__':
    main()
