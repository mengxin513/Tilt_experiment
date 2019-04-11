from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    with PdfPages('angle_calibration.pdf') as pdf:

        turns = np.array([0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2])
        cam_dis1 = np.array([61.9, 124.9, 190.2, 252.2, 318.8, 381.2, 445.7, 509.6])
        cam_dis2 = np.array([60.1, 124.5, 187.3, 252.5, 316.8, 381.4, 445.0, 507.1])
        cam_dis3 = np.array([62.0, 125.3, 189.3, 252.9, 317.1, 380.6, 446.0, 509.0])
        cam_dis = np.array((cam_dis1, cam_dis2, cam_dis3))

        angle_per_turn_in_mrad = 15

        ave_cam_dis = np.mean(cam_dis, axis = 0)
        angle = turns * angle_per_turn_in_mrad

        err = abs(cam_dis - ave_cam_dis[None, :])
        err = err.max(axis = 0)

        matplotlib.rcParams.update({'font.size': 12})

        fig, ax = plt.subplots(1, 1)

        ax.errorbar(ave_cam_dis, angle, xerr = err, fmt = 'ro')

        ax.set_xlabel(r'Camera displacement [$\mathrm{px}$]')
        ax.set_ylabel(r'Angle [$\mathrm{mrad}$]')

        coef = np.polyfit(ave_cam_dis, angle, 1)
        print('Angle per pixel in mrad: {}'.format(coef[0]))
        print('Offset: {}'.format(coef[1]))
        xp = np.linspace(0, 1.1 * max(ave_cam_dis), 100)
        p = np.poly1d(coef)
        ax.plot(xp, p(xp), 'b-')

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, ymax)

        plt.tight_layout()

        pdf.savefig(fig)
        plt.close(fig)

        regression = angle - p(ave_cam_dis)

        fig, ax = plt.subplots(1, 1)

        ax.plot(ave_cam_dis, regression, 'ro-')

        ax.set_xlabel(r'Average camera displacement [$\mathrm{px}$]')
        ax.set_ylabel(r'Regression [$\mathrm{mrad}$]')

        xmin, xmax = ax.get_xlim()
        ax.set_xlim(0, xmax)

        plt.tight_layout()

        pdf.savefig(fig)
        plt.close(fig)
