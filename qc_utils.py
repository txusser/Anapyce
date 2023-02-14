from os.path import exists
import nibabel as nib
import numpy as np


class qc_utils(object):

    def __init__(self):
        pass

    @staticmethod
    def _validation_png(img1_path, output_file):
        import matplotlib.pyplot as plt
        # Load the NIFTI images
        img1 = nib.load(img1_path)

        # Get the data from the images
        img1_data = img1.get_fdata()

        # Get the shape of the images
        shape = img1_data.shape
        n_slices = shape[2]
        slice_idx = [int(i * n_slices) for i in [0.25, 0.5, 0.75]]
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        for i, ax in enumerate(axs.flat):
            # Get the slices
            img1_slice = img1_data[:, :, slice_idx[i]]

            # Show the slices
            ax.imshow(img1_slice, cmap='gray', alpha=0.1)
            ax.set_title("Slice {}".format(slice_idx[i]))
        plt.show()

    @staticmethod
    def _overlay_png(img1_path, img2_path, output_file):
        import matplotlib.pyplot as plt
        # Load the NIFTI images
        img1 = nib.load(img1_path)
        img2 = nib.load(img2_path)

        # Get the data from the images
        img1_data = img1.get_fdata()
        img2_data = img2.get_fdata()

        # Get the shape of the images
        shape = img1_data.shape
        n_slices = shape[2]
        slice_idx = [int(i * n_slices) for i in [0.25, 0.5, 0.75]]
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        for i, ax in enumerate(axs.flat):
            # Get the slices
            img1_slice = img1_data[:, :, slice_idx[i]]
            img2_slice = img2_data[:, :, slice_idx[i]]

            # Show the slices
            ax.imshow(img1_slice, cmap='gray', alpha=0.7)
            ax.imshow(img2_slice, cmap='jet', alpha=0.3)
            ax.set_title("Slice {}".format(slice_idx[i]))
        plt.show()

    @staticmethod
    def _check_image_exists(input_image: str):
        if not exists(input_image):
            raise FileNotFoundError(f"{input_image} is not found.")

    @staticmethod
    def _check_input_image_shape(img_data):
        if img_data.ndim != 3:
            img_data = img_data[:, :, :, 0]
            return img_data

    @staticmethod
    def _check_input_image(img_path, output=False):

        from scipy.ndimage import center_of_mass

        # Load the Nifti image
        img = nib.load(img_path)
        img = nib.funcs.as_closest_canonical(img)

        # Get the image data and the affine matrix
        data = img.get_fdata()
        affine = img.affine

        # Calculate the center of mass of the image
        com = center_of_mass(data)

        # Compare the center of mass with the center of the image
        center = np.array(data.shape) / 2
        if not np.allclose(center, com):
            # Calculate the translation to move the center of mass to the center of the image
            translation = center - com

            # Update the affine matrix
            new_affine = affine.copy()
            new_affine[:3, 3] += translation

            # Create the new image with the updated affine matrix
            new_img = nib.Nifti1Image(data, new_affine)
        else:
            new_img = img

        if output:
            nib.save(new_img, output)
        else:
            nib.save(new_img, img_path)
