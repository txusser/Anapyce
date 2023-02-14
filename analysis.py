import os
import numpy as np
import nibabel as nib
from os.path import exists
import pandas as pd
import spm
import qc_utils


class analysis(object):

    def __init__(self, resources_path):
        pass

    @staticmethod
    def ncc(img1_path, img2_path):
        # Load Nifti images
        img1 = nib.load(img1_path)
        img2 = nib.load(img2_path)

        # Get image data as numpy arrays
        img1_data = img1.get_fdata()
        img2_data = img2.get_fdata()

        # Subtract the mean from each image
        img1_data = img1_data - np.mean(img1_data)
        img2_data = img2_data - np.mean(img2_data)

        # Calculate the numerator of the NCC equation
        numerator = np.sum(img1_data * img2_data)

        # Calculate the denominator of the NCC equation
        denominator = np.sqrt(np.sum(img1_data ** 2) * np.sum(img2_data ** 2))

        # Calculate and return the NCC
        return numerator / denominator

    @staticmethod
    def optimize_smoothing_group_ncc(spm_path, img_group_paths, template_path, sigma_range):

        from scipy.ndimage import gaussian_filter

        best_ncc = -np.inf
        best_sigma = None

        template = nib.load(template_path)
        template_data = np.nan_to_num(template.get_fdata())
        indexes = np.where(template_data > 0.1 * np.max(template_data))

        template_data = template_data - np.mean(template_data[indexes])

        for sigma in sigma_range:
            print("Testing Gaussian Filter with FHWM %s" % sigma)
            ncc_sum = 0
            ncc_count = 0

            spm_proc = spm.spm(spm_path)
            spm_proc.smooth_imgs(img_group_paths, [sigma, sigma, sigma])

            smooth_img_group = []

            for img_path in img_group_paths:
                components = os.path.split(img_path)
                simg_ = os.path.join(components[0], 's' + components[1])
                smooth_img_group.append(simg_)

            for img_path in smooth_img_group:
                # Load Nifti images
                img1 = nib.load(img_path)
                # Get image data as numpy arrays
                img1_data = np.nan_to_num(img1.get_fdata())
                # Subtract the mean from each image
                img1_data = img1_data - np.mean(img1_data[indexes])

                # Calculate the numerator of the NCC equation
                numerator = np.sum(img1_data[indexes] * template_data[indexes])
                # Calculate the denominator of the NCC equation
                denominator = np.sqrt(np.sum(img1_data[indexes] ** 2) * np.sum(template_data[indexes] ** 2))
                # Calculate the NCC
                ncc_value = numerator / denominator
                ncc_sum += ncc_value
                ncc_count += 1

            avg_ncc = ncc_sum / ncc_count
            # Check if this NCC value is the highest so far
            if avg_ncc > best_ncc:
                best_ncc = avg_ncc
                best_sigma = sigma
        return best_sigma, best_ncc

    @staticmethod
    def image_to_image_corr_atlas_based_spearsman(image_1, image_2, atlas):

        from scipy.stats import spearmanr

        # Load the two images + atlas using nibabel library
        img_1 = nib.load(image_1)
        img_1_data = img_1.get_fdata()

        img_2 = nib.load(image_2)
        img_2_data = img_2.get_fdata()

        atlas_img = nib.load(atlas)
        atlas_data = atlas_img.get_fdata()[:, :, :, 0]

        # Get the unique values of the atlas, which correspond to the ROIs
        rois = np.unique(atlas_data)

        # Initialize lists to store the mean values of the images for each ROI
        img_1_array = []
        img_2_array = []

        # Loop over the ROIs
        for i in rois:
            if i != 0:
                # Get the indices of the voxels that belong to the current ROI
                indx = np.where(atlas_data == i)

                # Calculate the mean value of the first image for the current ROI
                mean_1 = np.mean(img_1_data[indx])
                # Calculate the mean value of the second image for the current ROI
                mean_2 = np.mean(img_2_data[indx])

                # Append the mean values to the lists
                img_1_array.append(mean_1)
                img_2_array.append(mean_2)

        # Calculate the spearman correlation coefficient and p-value between the mean values of the images for each ROI
        rho, p = spearmanr(img_1_array, img_2_array)
        return rho, p

    @staticmethod
    def normalize_using_ref_region(input_image, output_image, ref_region):

        qc_utils.qc_utils._check_image_exists(input_image)

        pons_img = nib.load(ref_region).get_fdata()
        pons_img = pons_img[:, :, :, 0]
        pons_vox = np.where(pons_img == 1)

        input_img = nib.load(input_image)
        img_data = input_img.get_fdata()
        # img_data = qc_utils._check_input_image_shape(img_data)

        pons_value = np.mean(img_data[pons_vox])
        normalized_img = img_data / pons_value

        new_img = nib.Nifti1Image(normalized_img, input_img.affine, input_img.header)
        nib.save(new_img, output_image)

    @staticmethod
    def normalize_histogram(input_image, template, mask, output):

        fdg = nib.load(input_image)
        template = nib.load(template)
        mask = nib.load(mask)

        fdg_data = fdg.get_fdata()
        template_data = template.get_fdata()
        mask_data = mask.get_fdata()[:, :, :, 0]

        indx = np.where(mask_data == 1)
        mean_template = np.mean(template_data[indx])
        mean_fdg = np.mean(fdg_data[indx])

        fdg_data = fdg_data * (mean_template / mean_fdg)

        division = template_data[indx] / fdg_data[indx]
        values, bins = np.histogram(division, 250, range=[0, 3])
        amax = np.amax(values)
        indx = np.where(values == amax)
        norm_value = bins[indx][0]
        fdg_data = fdg_data * norm_value

        img = nib.Nifti1Image(fdg_data, fdg.affine, fdg.header)
        nib.save(img, output)

        return norm_value

    @staticmethod
    def create_atlas_csv(normals, output_csv, atlas_csv, atlas_hdr):

        atlas_df = pd.read_csv(atlas_csv)
        atlas_img = nib.load(atlas_hdr)
        atlas_data = atlas_img.get_fdata()[:, :, :, 0]

        for indx_, row_ in atlas_df.iterrows():

            roi_num = row_['ROI_NUM']
            roi_name = row_['ROI_NAME']
            index_ = np.where(atlas_data == roi_num)

            roi_values = []

            for img_ in normals:
                img_data = nib.load(img_).get_fdata()
                value = np.mean(img_data[index_])
                roi_values.append(value)

            roi_mean = np.mean(roi_values)
            roi_std = np.std(roi_values)

            atlas_df.loc[indx_, 'ROI_MEAN'] = roi_mean
            atlas_df.loc[indx_, 'ROI_STD'] = roi_std

            print('%s: %s +- %s' % (roi_name, roi_mean, roi_std))

        atlas_df.to_csv(output_csv)
        print('Done!')

    @staticmethod
    def create_atlas_csv_from_patsegs(normals, atlases, output_csv, atlas_csv):

        import math

        atlas_df = pd.read_csv(atlas_csv, sep=';')

        for indx_, row_ in atlas_df.iterrows():

            roi_num = row_['ROI_NUM']
            roi_name = row_['ROI_NAME']

            roi_values = []

            for i in range(len(normals)):
                img_ = normals[i]
                atlas_ = atlases[i]

                atlas_img = nib.load(atlas_)
                atlas_data = atlas_img.get_fdata()
                img_data = nib.load(img_).get_fdata()
                index_ = np.where(atlas_data == roi_num)
                if len(index_) > 0:
                    value = np.mean(img_data[index_])
                    roi_values.append(value)

            roi_values = [item for item in roi_values if not (math.isnan(item)) == True]
            roi_mean = np.mean(roi_values)
            roi_std = np.std(roi_values)

            atlas_df.loc[indx_, 'ROI_MEAN'] = roi_mean
            atlas_df.loc[indx_, 'ROI_STD'] = roi_std

            print('%s: %s +- %s' % (roi_name, str(round(roi_mean, 3)), str(round(roi_std, 3))))

        atlas_df.to_csv(output_csv)
        print('Done!')

    @staticmethod
    def transform_img_to_atlas_zscores(img_, out, atlas_csv, atlas_hdr):

        atlas_csv = atlas_csv
        atlas_df = pd.read_csv(atlas_csv)
        atlas_img = nib.load(atlas_hdr)
        atlas_data = atlas_img.get_fdata()

        if exists(img_):

            img = nib.load(img_)
            data = img.get_fdata()

            pat_atlas = np.zeros(atlas_data.shape)

            for indx, row in atlas_df.iterrows():
                roi_num = row['ROI_NUM']
                roi_mean = row['ROI_MEAN']
                roi_std = row['ROI_STD']

                index_ = np.where(atlas_data == roi_num)
                value = np.mean(data[index_])
                z_score = (value - roi_mean) / roi_std
                pat_atlas[index_] = z_score

            img = nib.Nifti1Image(pat_atlas, atlas_img.affine, atlas_img.header)
            nib.save(img, out)

            print('Done transforming %s to atlas z-scores!' % img_)

        else:
            raise FileNotFoundError("File not found at: " + img_)

    @staticmethod
    def create_mean_std_imgs(images, output_mean, output_std):

        """
        For a list of images, creates the mean and std images
        """

        sample_nii = images[0]
        sample_img = nib.load(sample_nii)
        sample_data = sample_img.get_fdata()

        mean_data = sample_data * 0
        std_data = sample_data * 0

        data = sample_data[:, :, :, np.newaxis]

        for i in range(1, len(images)):
            img = nib.load(images[i])
            img_data = img.get_fdata()
            img_data = img_data[:, :, :, np.newaxis]
            data = np.append(data, img_data, axis=3)

        for i in range(sample_data.shape[0]):
            for j in range(sample_data.shape[1]):
                for k in range(sample_data.shape[2]):
                    values = data[i, j, k, :]
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    mean_data[i, j, k] = mean_val
                    std_data[i, j, k] = std_val

        # print("Finished %s of %s slices." % (i, sample_data.shape[0]))

        mean_img = nib.AnalyzeImage(mean_data, sample_img.affine, sample_img.header)
        nib.save(mean_img, output_mean)
        std_img = nib.AnalyzeImage(std_data, sample_img.affine, sample_img.header)
        nib.save(std_img, output_std)
