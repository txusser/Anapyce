import nibabel as nib


class manip(object):

    def __init__(self, resources_path):
        pass

    @staticmethod
    def convert_to_analyze(nii_file):
        """This function converts a nifti file to an analyze file and saves it in the same directory.

        :param nii_file: the name of the nifti file to be converted
        :type nii_file: str
        :return: the name of the analyze file that was created
        :rtype: str
        """

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
        """This function applies a constant to an image using a specified operation, and saves the result in a new image.

        :param image: the name of the image file to be processed
        :type image: str
        :param c: the constant to be applied to the image
        :type c: float
        :param operation: the operation to be performed on the image. It can be 'mult', 'div' or 'sum'
        :type operation: str
        :param output: the name of the output image file. If False, the output file will have the same name as the input file
        :type output: str or bool
        :return: None
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