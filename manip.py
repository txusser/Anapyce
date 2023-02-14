import nibabel as nib


class manip(object):

    def __init__(self, resources_path):
        pass

    @staticmethod
    def convert_to_analyze(nii_file):
        # Load the NIFTI image using nibabel
        img = nib.load(nii_file)

        # Get the header information from the NIFTI image
        header = img.header

        # Get the image data from the NIFTI image
        data = img.get_fdata()

        # Save the Analyze image
        analyze_img = nib.analyze.AnalyzeImage(data, header.get_best_affine(), header)
        analyze_file = nii_file.split('.nii.gz')[0] + '.img'
        nib.save(analyze_img, analyze_file)
        print(f"File {nii_file} was converted to Analyze format and saved as {analyze_file}")
        return analyze_file

    @staticmethod
    def apply_constant_to_img(image, c, operation, output=False):
        """
        :param image:
        :param c:
        :param operation:
        :param output:
        """
        if output:
            output_file = output

        else:
            output_file = image

        img_ = nib.load(image)
        data = img_.get_fdata()

        if operation == 'mult':

            data = data*c

        elif operation == 'div':

            data = data/c

        elif operation == 'sum':

            data = data+c

        new_img = nib.AnalyzeImage(data, img_.affine, img_.header)
        nib.save(new_img, output_file)