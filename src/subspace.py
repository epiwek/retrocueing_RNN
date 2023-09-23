"""
This file contains classes used to fit a 3D subspace as well as find the best-fitting planes for a given dataset.

@author: emilia
"""
import numpy as np
from numpy.linalg import lstsq
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
import src.vec_operations as vops


class SinglePlaneSubspace:
    """
    Use this class to fit a 2D subspace (i.e., a plane) to data from a single cue location.
    """
    def __init__(self, coords_3d):
        self.coords_3d = coords_3d
        self.fitted_plane = None
        self.plane_vec1 = None
        self.plane_vec2 = None
        self.plane_vecs_xy = None
        self.plane_normal = None
        self.is_concave = None
        self.missing_vtx = None
        # self.plane_vecs_side_aligned = None

    def get_best_fit_plane(self):
        """
        Fit a plane to datapoints using PCA.

        Returns
        -------
        fitted_plane : sklearn.decomposition.PCA object
            fitted_plane.components_ gives the plane vectors

        """
        # center data
        data_centered = self.coords_3d - np.mean(self.coords_3d)

        # find plane of best fit using PCA from sklearn.decomposition
        pca = PCA(n_components=2)
        fitted_plane = pca.fit(data_centered)
        self.fitted_plane = fitted_plane
        self.plane_vec1 = fitted_plane.components_[0, :]
        self.plane_vec2 = fitted_plane.components_[1, :]
        self.plane_vecs_xy = fitted_plane.components_
        return self.fitted_plane

    def get_normal(self):
        """
        Get the plane normal.

        """
        if self.fitted_plane is None:
            self.get_best_fit_plane()

        self.plane_normal = np.cross(self.plane_vec1, self.plane_vec2)

    def get_new_basis(self):
        """
        Get the basis defined by the plane-defining vectors and the plane normal.
        :return: new_basis
        :rtype new_basis: np.ndarray
        """
        if self.plane_normal is None:
            self.get_normal()

        new_basis = np.stack((self.plane_vec1, self.plane_vec2, self.plane_normal), axis=0).T
        return new_basis

    def project_onto_plane(self):
        """
        Project the 3D datapoints onto the best fitting plane. Returns the 3D coordinates of the projections.
        :return: points_projected
        """
        if self.fitted_plane is None:
            self.get_best_fit_plane()

        plane_vecs = self.fitted_plane.components_
        n_points = self.coords_3d.shape[0]
        points_projected = np.stack([vops.get_projection(self.coords_3d[p, :], plane_vecs) for p in range(n_points)])
        return points_projected

    def compress_to_2d(self):
        """
        Compress the datapoints to 2D by projecting them onto the best fit plane and discarding the z-coordinate.
        Note the returned coordinates are expressed in the basis defined by the plane-defining vectors and the
        plane normal.

        """
        # project points onto the plane
        points_projected = self.project_onto_plane()

        # get new basis - defined by plane vectors and normal
        new_basis = self.get_new_basis()

        # change basis to get rid of the z-coords
        points_new_basis = (new_basis.T @ points_projected.T).T

        # test that change of basis got rid of the z-component
        assert points_new_basis[:, -1].round(6).sum() == 0, 'Change of basis failed to get rid of the z-component'

        return points_new_basis[:, :2]

    def check_is_concave(self):
        """
        Check if the quadrilateral formed by the datapoints is concave.

        """
        points_2d = self.compress_to_2d()

        # fit a convex hull to the datapoints
        hull = ConvexHull(points_2d)
        # check if it is a quadrilateral
        n_vertices = len(hull.vertices)
        self.missing_vtx = np.setdiff1d(np.arange(4), hull.vertices)
        self.is_concave = n_vertices < 4

    def detect_bowtie(self):
        """
        Detect a bowtie geometry.
 
        """
        # compress to 2D
        points_2d = self.compress_to_2d()

        # indices of the diagonal vertices
        diagonal_ixs = np.array([[0, 2], [1, 3]])
        # find the lines demarcated by the diagonals
        m1 = np.diff(points_2d[diagonal_ixs[0, :], 1]) / np.diff(points_2d[diagonal_ixs[0, :], 0])
        m2 = np.diff(points_2d[diagonal_ixs[1, :], 1]) / np.diff(points_2d[diagonal_ixs[1, :], 0])
        c1 = points_2d[diagonal_ixs[0, 0], 1] - m1 * points_2d[diagonal_ixs[0, 0], 0]
        c2 = points_2d[diagonal_ixs[1, 0], 1] - m2 * points_2d[diagonal_ixs[1, 0], 0]

        # find the point where they cross
        x = (c2 - c1) / (m1 - m2)
        y = m1 * x + c1

        # check if it lies between the ends of the two diagonals
        x_vs_d1 = np.logical_and(x >= points_2d[diagonal_ixs[0, :], 0].min(),
                                 x <= points_2d[diagonal_ixs[0, :], 0].max())
        y_vs_d1 = np.logical_and(y >= points_2d[diagonal_ixs[0, :], 1].min(),
                                 y <= points_2d[diagonal_ixs[0, :], 1].max())
        x_vs_d2 = np.logical_and(x >= points_2d[diagonal_ixs[1, :], 0].min(),
                                 x <= points_2d[diagonal_ixs[1, :], 0].max())
        y_vs_d2 = np.logical_and(y >= points_2d[diagonal_ixs[1, :], 1].min(),
                                 y <= points_2d[diagonal_ixs[1, :], 1].max())
        is_centre_within = np.all([x_vs_d1, x_vs_d2, y_vs_d1, y_vs_d2])

        # check if the datapoints can form a convex quadrilateral

        if self.is_concave is None:
            self.check_is_concave()

        is_bowtie = np.logical_and(not is_centre_within, not self.is_concave)

        return is_bowtie

    def construct_subspace(self):
        """Run the entire pipeline"""
        self.get_best_fit_plane()
        self.get_normal()
        self.check_is_concave()


class Geometry:
    """
    Use this class to fit a 3D subspace to data from a pair of conditions (e.g., for Cued geometry - cued colour
    representations for cued-upper and cued-lower trials). Data should contain only one timepoint.
    """
    def __init__(self, data, constants, subspace_labels=None):
        self.data = data  # shape (n_colours * n_locations, n_neurons)
        self.n_colours = constants.PARAMS['B']
        if subspace_labels is not None:
            self.subspace1_label = subspace_labels[0]
            self.subspace2_label = subspace_labels[1]
        self.PVEs = None
        self.coords_3d = None
        self.plane1_coords_3d = None
        self.plane2_coords_3d = None
        self.plane1 = None
        self.plane2 = None
        self.plane1_basis_corrected = None
        self.plane2_basis_corrected = None
        self.cos_theta = None
        self.theta_degrees = None
        self.psi_degrees = None
        self.bowtie1 = None
        self.bowtie2 = None
        self.reflection = None

    def get_3d_coords(self):
        """
        Run the first PCA with the data matrix to reduce data dimensionality from n_neurons to 3.
        """

        # Initialise PCA object
        pca = PCA(n_components=3)
        # demean and get coordinates in the reduced-dim space
        self.coords_3d = pca.fit_transform(self.data - self.data.mean())
        self.PVEs = pca.explained_variance_ratio_
        self.plane1_coords_3d = self.coords_3d[:self.n_colours, :]
        self.plane2_coords_3d = self.coords_3d[self.n_colours:, :]

    def get_best_fit_planes(self):
        """
        Fit planes to the location-specific colour coordinates.
        """
        if self.coords_3d is None:
            # run first PCA
            self.get_3d_coords()

        # get first plane
        self.plane1 = SinglePlaneSubspace(self.plane1_coords_3d)
        self.plane1.construct_subspace()
        # get second plane
        self.plane2 = SinglePlaneSubspace(self.plane2_coords_3d)
        self.plane2.construct_subspace()
        return self.plane1, self.plane2

    def pick_quadrilateral_sides(self):
        """
        Pick the corresponding sides of the quadrilateral as the basis for planes 1 and 2.
        :return: side_ixs
        """
        # get best fit planes
        if self.plane1 is None or self.plane2 is None:
            # run second PCA to find best fit planes
            self.get_best_fit_planes()

        # pick the sides of the quadrilateral that will be used as the plane basis vectors
        if np.any([self.plane1.is_concave, self.plane2.is_concave]):
            # need to pick the sides of the quadrilaterals that don't form the
            # concave angle to be the plane-defining vectors
            vtcs = \
                np.setdiff1d(np.arange(self.n_colours),
                             np.concatenate((self.plane1.missing_vtx, self.plane2.missing_vtx)))
            if len(vtcs) < 3:
                # raise ValueError('Both geometries are concave, and at mismatching vertices.')
                # revert to the standard choice for vertices
                vtcs = np.array([0, 1, 3])
        else:
            # pick the sides joining the 'red-yellow' and 'red-blue' sides
            vtcs = np.array([0, 1, 3])

        corner_ix = np.where(np.diff(vtcs).min() == 1)[0][0]  # index of the corner between the two quadrilateral sides
        side_ixs = [vtcs[corner_ix], vtcs[corner_ix + 1], vtcs[corner_ix], vtcs[corner_ix - 1]]
        return side_ixs

    def correct_plane_bases(self):
        """
        Set the plane basis vectors to correspond to sides of the quadrilateral as the basis for planes 1 and 2.
        """
        # pick corresponding sides of the quadrilateral
        side_ixs = self.pick_quadrilateral_sides()
        # indices of neighbouring sides of the parallelogram in the format: [corner vertex,side x,corner vertex,side y]

        # align the plane-defining vectors to the picked sides (but keep them orthogonal)
        points1 = self.plane1_coords_3d  # location 1 datapoints
        points2 = self.plane2_coords_3d  # location 2

        plane_vecs_aligned = []
        for points, plane in zip((points1, points2), (self.plane1, self.plane2)):
            # project datapoints onto the fitted plane

            plane_vecs = np.stack((plane.plane_vec1, plane.plane_vec2))

            points_proj = np.zeros(points.shape)
            com = np.mean(points, axis=0)  # centre of mass
            for i in range(points_proj.shape[0]):
                points_proj[i, :] = vops.get_projection(points[i, :] - com, plane_vecs)

            # get the vectors corresponding to the sides of the parallelogram
            # these will be the new plane-defining bases
            a = vops.get_vec_from_points(points_proj[side_ixs[0], :], points_proj[side_ixs[1], :])
            b = vops.get_vec_from_points(points_proj[side_ixs[2], :], points_proj[side_ixs[3], :])

            # normalise them
            a /= np.linalg.norm(a)
            b /= np.linalg.norm(b)

            # change basis to the one defined by the plane + its normal
            # this is so that the vector(s) can be rotated around the normal with
            # rotate_plane_by_angle(), which assumes that the z-component of the vector(s)
            # is 0

            plane_basis = np.concatenate((plane_vecs, plane.plane_normal[None, :]))
            a_new_basis = plane_basis @ a
            b_new_basis = plane_basis @ b

            # force the plane-defining vectors to be orthogonal
            if np.abs(np.dot(a, b)) > 0.001:
                # if they're not already - rotate vector a by to form a 90 degrees angle
                angle = np.arccos(np.dot(a_new_basis, b_new_basis))
                angle_diff = np.degrees(np.pi / 2 - angle)

                # rotate
                tmp = vops.rotate_plane_by_angle(b_new_basis, -angle_diff)
                if np.abs(np.dot(a_new_basis, tmp)) < 0.001:
                    b_new_basis = tmp
                else:
                    # rotate the other way
                    tmp = vops.rotate_plane_by_angle(b_new_basis, angle_diff)
                    # double check that vectors are orthogonal now
                    assert np.abs(np.dot(a_new_basis, tmp)) < 0.001, 'New vectors still not orthogonal'
                    b_new_basis = tmp

            # return to original (standard) basis
            b = plane_basis.T @ b_new_basis

            # double check that the new vectors in the standard basis *are* orthogonal
            assert np.abs(np.dot(a, b)) < 0.001, 'New vectors not orthogonal'
            plane_vecs_aligned.append(np.stack((a, b)))

        normal1_aligned = np.cross(plane_vecs_aligned[0][0, :], plane_vecs_aligned[0][1, :])
        normal2_aligned = np.cross(plane_vecs_aligned[1][0, :], plane_vecs_aligned[1][1, :])

        self.plane1_basis_corrected = np.concatenate((plane_vecs_aligned[0], normal1_aligned[None, :]), axis=0).T
        self.plane2_basis_corrected = np.concatenate((plane_vecs_aligned[1], normal2_aligned[None, :]), axis=0).T

    def get_cos_plane_angle_theta(self):
        """ Calculate the cosine of plane angle theta """
        # align the plane-defining vectors with respect to the sides of the quadrilateral
        self.correct_plane_bases()

        # calculate the angle between planes
        normal1 = self.plane1_basis_corrected[:, -1]
        normal2 = self.plane2_basis_corrected[:, -1]

        # since each normal has a length of 1, their dot product will be equal
        # to the cosine of the angle between them
        self.cos_theta = np.dot(normal1, normal2)

        return self.cos_theta, normal1, normal2

    def get_plane_angle_theta_sign(self):
        """ Get the sign of plane angle theta """
        _, normal1, normal2 = self.get_cos_plane_angle_theta()
        # define the sign of the angle
        # this is an arbitrary distinction, but necessary to do circular statistics
        # at the group level. Arc-cos will always map angles to the [0,180] range,
        # whereas we want them to span the full circle. This rectification
        # will also mean that the mean angle across group will never be 0.
        # Sign is determined based on the normal of plane 1 - if the z coordinate
        # is >= 0, it is positive, otherwise - negative.
        if normal1[-1] >= 0:
            sign_theta = 1
        else:
            sign_theta = -1

        return sign_theta

    def get_theta_degrees(self):
        """
        Get the signed angle between planes in degrees.
        """
        sign_theta = self.get_plane_angle_theta_sign()

        # get the plane angle - in degrees for convenience
        theta_degrees = np.degrees(np.arccos(self.cos_theta))
        theta_degrees *= sign_theta

        self.theta_degrees = theta_degrees
        return theta_degrees

    def make_coplanar(self):
        """
        Force the two planes two be coplanar, to calculate the phase alignment angle between the corresponding
        datapoints.
        """
        # points, normal1, normal2
        # get 3d coords and fitted planes
        if self.plane1_basis_corrected is None:
            self.correct_plane_bases()

        normal1 = self.plane1_basis_corrected[:, -1]
        normal2 = self.plane2_basis_corrected[:, -1]

        plane1_vecs = np.stack((self.plane1.plane_vec1, self.plane1.plane_vec2))
        plane2_vecs = np.stack((self.plane2.plane_vec1, self.plane2.plane_vec2))

        # center datapoints

        plane1_points = self.plane1.coords_3d - self.plane1.coords_3d.mean(0)
        plane2_points = self.plane2.coords_3d - self.plane2.coords_3d.mean(0)

        n_points = plane1_points.shape[0]

        # project the datapoints onto their corresponding planes
        plane1_points_p = np.stack([vops.get_projection(plane1_points[p, :], plane1_vecs) for p in range(n_points)])
        plane2_points_p = np.stack([vops.get_projection(plane2_points[p, :], plane2_vecs) for p in range(n_points)])

        # center so that mean of each plane is at the origin
        plane1_points_c = plane1_points_p - plane1_points_p.mean(0)
        plane2_points_c = plane2_points_p - plane2_points_p.mean(0)
        points_centered = np.concatenate((plane1_points_c, plane2_points_c), axis=0)
        # makes planes coplanar: by rotating plane 2 to be co-planar with plane 1

        # get the rotation axis and angle
        # if planes are already co-planar, leave them as they are - need a workaround because cross-product is not
        # defined then (i.e. it's 0)
        if np.abs(np.dot(normal1, normal2).round(6)) == 1:
            R = np.eye(len(normal1))  # rotation matrix
        else:
            # calculate the rotation angle (i.e. angle between the two normals)
            rot_angle = np.arccos(np.dot(normal1, normal2))
            if rot_angle > np.pi / 2:
                rot_angle += np.pi

            # get the rotation axis
            rot_axis = np.cross(normal1, normal2) / np.linalg.norm(np.cross(normal1, normal2))

            # construct the rotation matrix
            R = vops.construct_rot_matrix(rot_axis, rot_angle)

        # apply to plane2 points
        plane2_points = R.T @ points_centered[n_points:, :].T

        # get new coordinates
        points_coplanar = np.concatenate((points_centered[:n_points, :], plane2_points.T), axis=0)

        return points_coplanar

    def get_phase_alignment(self):
        """
        Calculate the phase alignment angle psi (without correcting for orthogonal planes).
        """
        # rotate plane 2 so that it is parallel to plane 1
        points_coplanar = self.make_coplanar()

        # get rid of the z-coordinate (change basis to that defined by plane 1)
        plane1 = SinglePlaneSubspace(points_coplanar[:self.n_colours, :])
        plane2 = SinglePlaneSubspace(points_coplanar[self.n_colours:, :])

        plane1.plane_vec1 = self.plane1.plane_vec1
        plane1.plane_vec2 = self.plane1.plane_vec2
        plane1.plane_normal = np.cross(plane1.plane_vec1, plane1.plane_vec2)
        plane1_2d = plane1.compress_to_2d()

        # set the basis to that of plane 1 - want to project the plane 2 datapoints into the basis of plane 1
        plane2.plane_vec1 = self.plane1.plane_vec1
        plane2.plane_vec2 = self.plane1.plane_vec2
        plane2.plane_normal = np.cross(plane2.plane_vec1, plane2.plane_vec2)
        plane2.fitted_plane = plane1.fitted_plane

        plane2_2d = plane2.compress_to_2d()

        # calculate the phase-alignment using procrustes analysis
        plane1_std, plane2_std, disparity, R, s = vops.procrustes(plane1_2d, plane2_2d)

        self.reflection = np.linalg.det(R) < 0
        self.bowtie1 = plane1.detect_bowtie()
        # vops.detect_bowtie(plane1_2d)
        self.bowtie2 = plane2.detect_bowtie()
        if np.any([self.reflection, self.bowtie1, self.bowtie2]):
            # estimating phase alignment doesn't make sense
            pa = np.nan
        else:
            pa = -np.degrees(np.arctan2(R[1, 0], R[0, 0]))

        return pa

    def get_psi_degrees(self):
        """ Get the phase alignment angle psi in degrees. If planes are orthogonal, psi is set to NaN."""
        cos_theta, normal1, normal2 = self.get_cos_plane_angle_theta()

        # calculate phase alignment
        # if angle between planes is within +-[90,180] range, it means that the planes are mirror images and
        # calculating phase alignment does not make sense - set pa to nan
        if cos_theta <= 0:
            self.psi_degrees = np.nan
        else:
            self.psi_degrees = self.get_phase_alignment()

        return self.psi_degrees

    def get_geometry(self):
        """Run the entire pipeline and get the geometry measures: plane angle theta and phase alignment angle psi."""
        self.get_3d_coords()
        self.get_best_fit_planes()
        self.correct_plane_bases()
        self.get_theta_degrees()
        self.get_psi_degrees()
