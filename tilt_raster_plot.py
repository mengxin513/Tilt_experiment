import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages

if __name__ == "__main__":

    with PdfPages('tilt_raster_standard.pdf') as pdf:

        print ('Loading data...')

        angle_per_pixel_in_mrad = 0.0586
        df = h5py.File('tilt_raster.hdf5', mode = 'r')
        for i in range(len(df)):
            group = df['raster%03d' % i]
            subgroup = group['standard_raster000']
            n = len(subgroup)
            data = np.zeros([n, 6])
            for j in range(n):
                dset = subgroup['data%03d' % j]
                data[j, 0:3] = dset['cam_position']
                data[j, 3:6] = dset['stage_position']

            t = data[:, 0]
            pixel_shifts = np.zeros([n, 3])
            location_shifts = np.zeros([n, 3])

            for i in range(n):
                pixel_shifts[i, 0:2] = data[i, 1:3] - data[int((n - 1) / 2), 1:3]
                pixel_shifts[i, 2] = 1
                location_shifts[i, 0] = data[i, 3]
                location_shifts[i, 1] = data[i, 5]
                location_shifts[i, 2] = 1

            A, res, rank, s = np.linalg.lstsq(location_shifts, pixel_shifts)
            #A is the least squares solution pixcel_shifts*A = location_shifts
            #res is the sums of residuals location_shifts - pixcel_shifts*A
            #rank is rank of matrix pixcel_shifts
            #s is singular values of pixcel_shifts
            print(A)

            transformed_location_shifts = np.dot(location_shifts, A)

            matplotlib.rcParams.update({'font.size': 12})

            fig, ax = plt.subplots(1, 1)
            graph = ax.quiver(location_shifts[:, 0] * 0.00960, location_shifts[:, 1] * 0.00772, pixel_shifts[:, 0] * angle_per_pixel_in_mrad, pixel_shifts[:, 1] * angle_per_pixel_in_mrad)
            legend = ax.quiverkey(graph, X = 0.3, Y = 1.1, U = 1, label = '1 mrad', labelpos = 'E')
            ax.set_xlabel(r'Stage X Coordinate [$\mathrm{\mu m}$]')
            ax.set_ylabel(r'Stage Y Coordinate [$\mathrm{\mu m}$]')
            
            plt.show(fig)

            pdf.savefig(fig, bbox_inches = "tight", bbox_extra_artists = [legend])
            plt.close(fig)

            fig, ax = plt.subplots(1, 1)
            ax.plot(location_shifts[:, 0] * 0.00960, location_shifts[:, 1] * 0.00772, 'r+')
            ax.set_xlabel(r'Stage X coordinate [$\mathrm{\mu m}$]')
            ax.set_ylabel(r'Stage Y coordinate [$\mathrm{\mu m}$]')

            plt.tight_layout()

            pdf.savefig(fig)
            plt.close(fig)

            fig, ax = plt.subplots(1, 1)
            line1, = ax.plot(transformed_location_shifts[:, 0] * angle_per_pixel_in_mrad, transformed_location_shifts[:, 1] * angle_per_pixel_in_mrad, 'r+')
            line2, = ax.plot(pixel_shifts[:, 0] * angle_per_pixel_in_mrad, pixel_shifts[:, 1] * angle_per_pixel_in_mrad, 'b+')

            ax.legend((line1, line2), ('Transformed location shifts', 'Measured angular shifts'))

            ax.set_xlabel(r'Transformaed stage X coordinate [$\mathrm{mrad}$]')
            ax.set_ylabel(r'Transformed stage Y coordinate [$\mathrm{mrad}$]')

            plt.tight_layout()

            pdf.savefig(fig)
            plt.close(fig)
