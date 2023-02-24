import os
import shutil
import numpy as np
import nibabel as nib
from os.path import join, exists, dirname
from qc_utils import qc_utils


class spm(object):

    def __init__(self, spm_exe):

        self.spm_path = spm_exe
        if not exists(self.spm_path):
            raise FileNotFoundError(f"{self.spm_path} is not found.")

    def normalize_pet(self, image_to_norm, template_image, images_to_write=False,
                      bb=None, write_vox_size='[1 1 1]', wrapping=True, interpolation=4):

        if bb is None:
            bb = [-90, -126, -72, 90, 90, 108]

        source_img_path, source_img_name = os.path.split(image_to_norm)
        # Set the output file name
        mfile_name = join(source_img_path, 'normalize.m')

        design_type = "matlabbatch{1}.spm.tools.oldnorm.estwrite."

        if not images_to_write:
            images_to_write = [image_to_norm]

        new_spm = open(mfile_name, "a")

        new_spm.write(
            design_type + "subj.source = {'" + image_to_norm + ",1'}\n;" +
            design_type + "subj.wtsrc = '';" + "\n" +
            design_type + "subj.resample = {"
        )

        for image in images_to_write:
            new_spm.write("'" + image + ",1'" + "\n")
        new_spm.write("};" + "\n")

        new_spm.write(
            design_type + "eoptions.template = {'" + template_image + ",1'};\n" +
            design_type + "eoptions.weight = '';\n" +
            design_type + "eoptions.smosrc = 8;\n" +
            design_type + "eoptions.smoref = 3;\n" +
            design_type + "eoptions.regtype ='mni';\n" +
            design_type + "eoptions.cutoff = 15;\n" +
            design_type + "eoptions.nits = 16;\n" +
            design_type + "eoptions.reg = 1;\n" +
            design_type + "roptions.preserve = 0;\n" +
            design_type + "roptions.bb =[" + str(bb[0]) + " " + str(bb[1]) + " " + str(bb[2]) + "\n" +
            str(bb[3]) + " " + str(bb[4]) + " " + str(bb[5]) + "];" + "\n" +
            design_type + "roptions.vox =" + write_vox_size + ";" + "\n" +
            design_type + "roptions.interp =" + str(interpolation) + ";" + "\n")

        if wrapping:
            new_spm.write(design_type + "roptions.wrap = [1 1 1];" + "\n")
        else:
            new_spm.write(design_type + "roptions.wrap = [0 0 0];" + "\n")

        new_spm.write(design_type + "roptions.prefix ='w';\n")
        new_spm.close()

        os.system('%s run %s' % (self.spm_path, mfile_name))

        components = os.path.split(images_to_write[0])
        output = os.path.join(components[0], 'w' + components[1])

        transformation_matrix = image_to_norm[0:-4] + "_sn.mat"

        return output, transformation_matrix

    def normalize_mri(self, image_to_norm, template_image, images_to_write=False,
                      bb=None, write_vox_size='[1 1 1]', interpolation=4):

        if bb is None:
            bb = [-90, -126, -72, 90, 90, 108]
        source_img_path, source_img_name = os.path.split(image_to_norm)
        # Set the output file name
        mfile_name = join(source_img_path, 'normalize.m')

        design_type = "matlabbatch{1}.spm.spatial.normalise.estwrite."

        if not images_to_write:
            images_to_write = [image_to_norm]

        new_spm = open(mfile_name, "w")

        new_spm.write(
            design_type + "subj.vol = {'" + image_to_norm + ",1'};" + "\n" +
            design_type + "subj.resample = {" + "\n"
        )

        for image in images_to_write:
            new_spm.write("'" + image + ",1'" + "\n")
        new_spm.write("};" + "\n")

        new_spm.write(
            design_type + "eoptions.biasreg = 0.01;\n" +
            design_type + "eoptions.biasfwhm = 60;\n" +
            design_type + "eoptions.tpm = {'" + template_image + "'};\n" +
            design_type + "eoptions.affreg = 'mni';\n" +
            design_type + "eoptions.reg = [0 0.001 0.5 0.05 0.2];\n" +
            design_type + "eoptions.fwhm = 0;\n" +
            design_type + "eoptions.samp = 3;\n" +
            design_type + "woptions.bb = [" +
            str(bb[0]) + " " + str(bb[1]) + " " + str(bb[2]) + "\n" +
            str(bb[3]) + " " + str(bb[4]) + " " + str(bb[5]) + "];" + "\n" +
            design_type + "woptions.vox = " + write_vox_size + ";" + "\n" +
            design_type + "woptions.interp = " + str(interpolation) + ";" + "\n")

        new_spm.close()

        os.system('%s run %s' % (self.spm_path, mfile_name))

        components_1 = os.path.split(images_to_write[0])
        output_1 = os.path.join(components_1[0], 'w' + components_1[1])

        components_2 = os.path.split(images_to_write[1])
        output_2 = os.path.join(components_2[0], 'w' + components_2[1])

        output_png = os.path.join(components_1[0], "normalization.png")
        qc_utils._overlay_png(output_1, output_2, output_png)

        matrix_name = "y_" + os.path.basename(image_to_norm)[0:-3] + "nii"
        transformation_matrix = os.path.join(components_1[0], matrix_name)

        return output_1, transformation_matrix

    def new_deformations(self, def_matrix, images_to_deform, interpolation, prefix='w'):

        source_img_path, source_img_name = os.path.split(images_to_deform[0])
        # Set the output file name
        mfile_name = join(source_img_path, 'deformations.m')

        design_type_comp = "matlabbatch{1}.spm.util.defs.comp{1}."
        design_type_out = "matlabbatch{1}.spm.util.defs.out{1}."

        new_spm = open(mfile_name, "w")

        new_spm.write(
            design_type_comp + "def = {'" + def_matrix + "'};\n" +
            design_type_out + "pull.fnames = {" + "\n"
        )

        for image in images_to_deform:
            new_spm.write("'" + image + "'\n")
        new_spm.write("};\n")

        new_spm.write(
            design_type_out + "pull.savedir.savesrc = 1;\n" +
            design_type_out + "pull.interp =" + str(interpolation) + ";\n" +
            design_type_out + "pull.mask = 0;\n" +
            design_type_out + "pull.fwhm = [0 0 0];\n" +
            design_type_out + "pull.prefix ='" + prefix + "';\n"
        )

        new_spm.close()

        os.system('%s run %s' % (self.spm_path, mfile_name))

    def old_deformations(self, def_matrix, base_image, images_to_deform, interpolation):

        source_img_path, source_img_name = os.path.split(images_to_deform[0])
        # Set the output file name
        mfile_name = join(source_img_path, 'deformations.m')

        design_type_comp = "matlabbatch{1}.spm.util.defs.comp{1}.inv."
        design_type_out = "matlabbatch{1}.spm.util.defs.out{1}."

        new_spm = open(mfile_name, "w")

        new_spm.write(
            design_type_comp + "comp{1}.sn2def.matname = {'" + def_matrix + "'};" + "\n" +
            design_type_comp + "comp{1}.sn2def.vox = [NaN NaN NaN];" + "\n" +
            design_type_comp + "comp{1}.sn2def.bb = [NaN NaN NaN" + "\n" +
            "NaN NaN NaN];" + "\n" +
            design_type_comp + "space = {'" + base_image + "'};" + "\n" +
            design_type_out + "pull.fnames = {" + "\n"
        )

        for image in images_to_deform:
            new_spm.write("'" + image + "'" + "\n")
        new_spm.write("};" + "\n")

        new_spm.write(design_type_out + "pull.savedir.saveusr = {'" + source_img_path + "/'};" + "\n" +
                      design_type_out + "pull.interp = " + str(interpolation) + ";" + "\n" +
                      design_type_out + "pull.mask = 1;\n" +
                      design_type_out + "pull.fwhm = [0 0 0];\n"
                      )
        new_spm.close()

        os.system('%s run %s' % (self.spm_path, mfile_name))

    def apply_normalization_to_atlas(self, def_matrix, norm_mri, fs_atlas):

        source_img_path, source_img_name = os.path.split(fs_atlas)
        # Set the output file name
        mfile_name = join(source_img_path, 'deformations.m')

        design_type = 'matlabbatch{1}.spm.util.defs.'

        new_spm = open(mfile_name, "w")
        new_spm.write(design_type + "comp{1}.inv.comp{1}.def = {'" + def_matrix + "'};\n")
        new_spm.write(design_type + "comp{1}.inv.space = {'" + norm_mri + "'};\n")
        new_spm.write(design_type + "out{1}.push.fnames = {'" + fs_atlas + "'};\n")
        new_spm.write(design_type + "out{1}.push.weight = {''};\n")
        new_spm.write(design_type + "out{1}.push.savedir.savesrc = 1;\n")
        new_spm.write(design_type + "out{1}.push.fov.file = {'" + norm_mri + "'};\n")
        new_spm.write(design_type + "out{1}.push.preserve = 2;\n")
        new_spm.write(design_type + "out{1}.push.fwhm = [0 0 0];\n")
        new_spm.write(design_type + "out{1}.push.prefix = 'w';\n")
        new_spm.close()

        os.system('%s run %s' % (self.spm_path, mfile_name))

        components = os.path.split(fs_atlas)
        output = os.path.join(components[0], "w" + components[1])

        output_png = os.path.join(components[0], "deformations.png")
        qc_utils._overlay_png(norm_mri, output, output_png)

    def coregister(self, reference_image, source_image):

        source_img_path, source_img_name = os.path.split(source_image)
        # Set the output file name
        mfile_name = join(source_img_path, 'coregister.m')

        design_type = "matlabbatch{1}.spm.spatial.coreg.estwrite."

        new_spm = open(mfile_name, "w")

        new_spm.write(design_type + "ref = {'" + reference_image + ",1'};\n")
        new_spm.write(design_type + "source = {'" + source_image + ",1'};\n")
        new_spm.write(design_type + "other = {''};\n")
        new_spm.write(design_type + "eoptions.cost_fun = 'nmi';\n")
        new_spm.write(design_type + "eoptions.sep = [4 2];\n")
        new_spm.write(
            design_type + "eoptions.tol = [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001];\n")
        new_spm.write(design_type + "eoptions.fwhm = [7 7]\n;")
        new_spm.write(design_type + "roptions.interp = 4;\n")
        new_spm.write(design_type + "roptions.wrap = [0 0 0];\n")
        new_spm.write(design_type + "roptions.mask = 0;\n")
        new_spm.write(design_type + "roptions.prefix = 'r';\n")
        new_spm.close()

        os.system('%s run %s' % (self.spm_path, mfile_name))

        components = os.path.split(source_image)
        output = os.path.join(components[0], "r" + components[1])

        output_png = os.path.join(components[0], "coregister.png")
        qc_utils._overlay_png(reference_image, output, output_png)

        return output

    def run_2sample_ttest(self, save_dir, group1, group1_ages, group2, group2_ages,
                          mask, dependence=0, variance=1, gmscaling=0, ancova=0, global_norm=1,
                          contrast_name='contrast', contrast='[1 -1 0]'):

        if exists(save_dir):
            shutil.rmtree(save_dir)

        os.makedirs(save_dir)

        print('Creating SPM model....')

        mfile_model = join(save_dir, 'model.m')

        self.create_mfile_model_pet_with_1_cov(mfile_model, save_dir, group1, group1_ages, group2, group2_ages,
                                               mask, dependence, variance, gmscaling, ancova, global_norm)

        print('%s run %s' % (self.spm_path, mfile_model))
        os.system('%s run %s' % (self.spm_path, mfile_model))

        print('Estimating model....')

        mfile_estimate = join(save_dir, 'estimate.m')
        spm_mat = join(save_dir, 'SPM.mat')

        self.create_mfile_estimate_model(mfile_estimate, spm_mat)
        os.system('%s run %s' % (self.spm_path, mfile_estimate))

        print('Calculating results....')

        mfile_results = join(save_dir, 'results.m')
        spm_mat = join(save_dir, 'SPM.mat')

        self.create_mfile_contrast(mfile_results, spm_mat, contrast_name=contrast_name, contrast=contrast)
        os.system('%s run %s' % (self.spm_path, mfile_results))

        print('Converting results to Cohens d....')

        out_t_values = join(save_dir, 'spmT_0001.nii')
        out_cohens = join(save_dir, 'cohens_d.nii')
        self.spm_map_2_cohens_d(out_t_values, out_cohens, len(group1), len(group2))

        print('Calculating thresholds for Cohens d (FDR corrected ....')
        self.get_tvalue_thresholds_FDR(out_t_values, len(group1), len(group2))

    @staticmethod
    def create_mfile_normalize_multiple_pets(mfile, images_to_norm, template_image, cutoff=15, nits=16, reg=1,
                                             preserve=0, affine_regularization_type='mni', source_image_smoothing=8,
                                             template_image_smoothing=3, bb=None,
                                             write_vox_size='[1 1 1]', wrapping=True, interpolation=4, prefix='w'):

        """
        This function creates a mfile to later run with MATLAB.
        mfile: Destination of the created mfile
        images_to_norm = A list of all the images you want to normalize (Nifti (.nii) of Analyze (.img)).
        template_image = Template image to normalize the images to
        rest of the parameters = Consult SPM
        """
        if bb is None:
            bb = [-91, -127, -73, 90, 90, 108]

        design_type = "matlabbatch{1}.spm.tools.oldnorm.estwrite."

        new_spm = open(mfile, "a")

        for i in range(len(images_to_norm)):
            new_spm.write(
                design_type + "subj(" + str(i + 1) + ").source = {'" + images_to_norm[i] + ",1'};" + "\n" +
                design_type + "subj(" + str(i + 1) + ").wtsrc = '';" + "\n" +
                design_type + "subj(" + str(i + 1) + ").resample = {'" + images_to_norm[i] + ",1'};" + "\n")

        new_spm.write(
            design_type + "eoptions.template = {'" + template_image + ",1'};" + "\n" +
            design_type + "eoptions.weight = '';" + "\n" +
            design_type + "eoptions.smosrc =" + str(source_image_smoothing) + ";" + "\n" +
            design_type + "eoptions.smoref =" + str(template_image_smoothing) + ";" + "\n" +
            design_type + "eoptions.regtype ='" + affine_regularization_type + "';" + "\n" +
            design_type + "eoptions.cutoff =" + str(cutoff) + ";" + "\n" +
            design_type + "eoptions.nits =" + str(nits) + ";" + "\n" +
            design_type + "eoptions.reg =" + str(reg) + ";" + "\n" +
            design_type + "roptions.preserve =" + str(preserve) + ";" + "\n" +
            design_type + "roptions.bb =[" + str(bb[0]) + " " + str(bb[1]) + " " + str(bb[2]) + "\n" +
            str(bb[3]) + " " + str(bb[4]) + " " + str(bb[5]) + "];" + "\n" +
            design_type + "roptions.vox =" + write_vox_size + ";" + "\n" +
            design_type + "roptions.interp =" + str(interpolation) + ";" + "\n")

        if wrapping:
            new_spm.write(design_type + "roptions.wrap = [1 1 1];" + "\n")
        else:
            new_spm.write(design_type + "roptions.wrap = [0 0 0];" + "\n")

        new_spm.write(design_type + "roptions.prefix ='" + prefix + "';" + "\n")
        new_spm.write("spm('defaults', 'PET');\n")
        new_spm.write("spm_jobman('initcfg');\n")
        new_spm.write("spm_jobman('run',matlabbatch);\n")
        new_spm.close()

    def smooth_imgs(self, images_to_smooth, smoothing):

        source_img_path, source_img_name = os.path.split(images_to_smooth[0])
        # Set the output file name
        mfile_name = 'smooth.m'

        design_type = "matlabbatch{1}.spm.spatial.smooth."
        smoothing_array = "[" + str(smoothing[0]) + " " + str(smoothing[1]) + " " + str(smoothing[2]) + "]"

        new_spm = open(mfile_name, "w")
        new_spm.write(design_type + "data = {\n")

        for i in images_to_smooth:
            new_spm.write("'" + i + ",1'\n")

        new_spm.write("};" + "\n")
        new_spm.write(design_type + "fwhm =" + smoothing_array + ";" + "\n")
        new_spm.write(design_type + "dtype = 0;" + "\n")
        new_spm.write(design_type + "im = 0;" + "\n")
        new_spm.write(design_type + "prefix ='" + 's' + "';" + "\n")
        new_spm.write("spm('defaults', 'PET');\n")
        new_spm.write("spm_jobman('initcfg');\n")
        new_spm.write("spm_jobman('run',matlabbatch);\n")
        new_spm.close()

        os.system('%s run %s' % (self.spm_path, mfile_name))

    def create_mfile_cat12seg_multiple_imgs(self, mfile, images_to_seg, template_tpm, template_volumes,
                                            number_of_cores=4, biasacc=0.5, APP=1070, kamap=0, LASstr=0.5,
                                            gcutstr=2, WMHC=1, regstr=0.5, output_vox_size=1,
                                            restypes_optimal="[1 0.1]", out_surf=1, out_surf_measure=1,
                                            atlas_nm=1, atlas_lbpa=0, atlas_cobra=0, atlas_hammers=0,
                                            atlas_custom=False,
                                            gm_native=0, gm_modul=1, gm_dartel=0, wm_native=0, wm_modul=1, wm_dartel=0,
                                            csf_native=0, csf_modul=0, csf_warped=0, csf_dartel=0, ct_native=0,
                                            ct_warped=0, ct_dartel=0,
                                            pp_native=0, pp_warped=0, pp_dartel=0, wmh_native=0, wmh_modul=0,
                                            wmh_warped=0, wmh_dartel=0,
                                            sl_native=0, sl_modul=0, sl_warped=0, sl_dartel=0, tpmc_native=0,
                                            tpmc_modul=0, tpmc_warped=0, tpmc_dartel=0,
                                            atlas_native=0, atlas_warped=0, atlas_dartel=0, labels_native=1,
                                            labels_warped=0, labels_dartel=0,
                                            bias_warped=1, las_native=0, las_warped=0, las_dartel=0, jacobian_warped=0,
                                            output_warps="[1 0]"):

        """
        This function creates a mfile to later run with MATLAB.
        mfile: Destination of the created mfile
        images_to_seg = A list of all the images you want to segment (Nifti (.nii) of Analyze (.img)).
        template_tpm = Template image to normalize the images to (something like ... PATH_TO/spm12/tpm/TPM.nii)
        template_volumes = Templete volumes image from CAT12 (something like ... PATH_TO/spm12/toolbox/cat12/template_volumes/Template_0_IXI555_MNI152_GS.nii)
        rest of the parameters = Consult CAT12 segment
        """

        design_type = "matlabbatch{1}.spm.tools.cat.estwrite."

        new_spm = open(mfile, 'w')

        new_spm.write(design_type + "data = {\n")
        for i in range(len(images_to_seg)):
            new_spm.write("'" + images_to_seg[i] + ",1'\n")
        new_spm.write("};" + "\n")

        new_spm.write(design_type + "data_wmh = {''};" + "\n")
        new_spm.write(design_type + "nproc = " + str(number_of_cores) + ";\n")
        new_spm.write(design_type + "useprior = '';" + "\n")
        new_spm.write(design_type + "opts.tpm = {'" + template_tpm + "'};\n")
        new_spm.write(design_type + "useprior = '';" + "\n")
        new_spm.write(design_type + "opts.affreg = 'mni';" + "\n")
        new_spm.write(design_type + "opts.biasacc = " + str(biasacc) + ";\n")

        new_spm.write(design_type + "extopts.APP = " + str(APP) + ";\n")
        new_spm.write(design_type + "extopts.spm_kamap = " + str(kamap) + ";\n")
        new_spm.write(design_type + "extopts.LASstr = " + str(LASstr) + ";\n")
        new_spm.write(design_type + "extopts.gcutstr = " + str(gcutstr) + ";\n")
        new_spm.write(design_type + "extopts.WMHC = " + str(WMHC) + ";\n")
        new_spm.write(design_type + "extopts.registration.shooting.shootingtpm = {'" + template_volumes + "'};\n")
        new_spm.write(design_type + "extopts.registration.shooting.regstr = " + str(regstr) + ";\n")
        new_spm.write(design_type + "extopts.vox = " + str(output_vox_size) + ";\n")
        new_spm.write(design_type + "extopts.restypes.optimal = " + restypes_optimal + ";\n")
        new_spm.write(design_type + "extopts.ignoreErrors = 1;\n")

        design_outputs = design_type + "output."

        new_spm.write(design_outputs + "surface = " + str(out_surf) + ";\n")
        new_spm.write(design_outputs + "surf_measures = " + str(out_surf_measure) + ";\n")
        new_spm.write(design_outputs + "ROImenu.atlases.neuromorphometrics = " + str(atlas_nm) + ";\n")
        new_spm.write(design_outputs + "ROImenu.atlases.lpba40 = " + str(atlas_lbpa) + ";\n")
        new_spm.write(design_outputs + "ROImenu.atlases.cobra = " + str(atlas_cobra) + ";\n")
        new_spm.write(design_outputs + "ROImenu.atlases.hammers = " + str(atlas_hammers) + ";\n")
        if atlas_custom == False:
            new_spm.write(design_outputs + "ROImenu.atlases.ownatlas = {''};\n")
        else:
            new_spm.write(design_outputs + "ROImenu.atlases.ownatlas = {'" + atlas_custom + "'};\n")

        new_spm.write(design_outputs + "GM.native = " + str(gm_native) + ";\n")
        new_spm.write(design_outputs + "GM.mod = " + str(gm_modul) + ";\n")
        new_spm.write(design_outputs + "GM.dartel = " + str(gm_dartel) + ";\n")

        new_spm.write(design_outputs + "WM.native = " + str(wm_native) + ";\n")
        new_spm.write(design_outputs + "WM.mod = " + str(wm_modul) + ";\n")
        new_spm.write(design_outputs + "WM.dartel = " + str(wm_dartel) + ";\n")

        new_spm.write(design_outputs + "CSF.native = " + str(csf_native) + ";\n")
        new_spm.write(design_outputs + "CSF.warped = " + str(csf_warped) + ";\n")
        new_spm.write(design_outputs + "CSF.mod = " + str(csf_modul) + ";\n")
        new_spm.write(design_outputs + "CSF.dartel = " + str(csf_dartel) + ";\n")

        new_spm.write(design_outputs + "ct.native = " + str(ct_native) + ";\n")
        new_spm.write(design_outputs + "ct.warped = " + str(ct_warped) + ";\n")
        new_spm.write(design_outputs + "ct.dartel = " + str(ct_dartel) + ";\n")

        new_spm.write(design_outputs + "pp.native = " + str(pp_native) + ";\n")
        new_spm.write(design_outputs + "pp.warped = " + str(pp_warped) + ";\n")
        new_spm.write(design_outputs + "pp.dartel = " + str(pp_dartel) + ";\n")

        new_spm.write(design_outputs + "WMH.native = " + str(wmh_native) + ";\n")
        new_spm.write(design_outputs + "WMH.warped = " + str(wmh_warped) + ";\n")
        new_spm.write(design_outputs + "WMH.mod = " + str(wmh_modul) + ";\n")
        new_spm.write(design_outputs + "WMH.dartel = " + str(wmh_dartel) + ";\n")

        new_spm.write(design_outputs + "SL.native = " + str(sl_native) + ";\n")
        new_spm.write(design_outputs + "SL.warped = " + str(sl_warped) + ";\n")
        new_spm.write(design_outputs + "SL.mod = " + str(sl_modul) + ";\n")
        new_spm.write(design_outputs + "SL.dartel = " + str(sl_dartel) + ";\n")

        new_spm.write(design_outputs + "TPMC.native = " + str(tpmc_native) + ";\n")
        new_spm.write(design_outputs + "TPMC.warped = " + str(tpmc_warped) + ";\n")
        new_spm.write(design_outputs + "output.TPMC.mod = " + str(tpmc_modul) + ";\n")
        new_spm.write(design_outputs + "TPMC.dartel = " + str(tpmc_dartel) + ";\n")

        new_spm.write(design_outputs + "atlas.native = " + str(atlas_native) + ";\n")
        new_spm.write(design_outputs + "atlas.warped = " + str(atlas_warped) + ";\n")
        new_spm.write(design_outputs + "atlas.dartel = " + str(atlas_dartel) + ";\n")

        new_spm.write(design_outputs + "label.native = " + str(labels_native) + ";\n")
        new_spm.write(design_outputs + "label.warped = " + str(labels_warped) + ";\n")
        new_spm.write(design_outputs + "label.dartel = " + str(labels_dartel) + ";\n")
        new_spm.write(design_outputs + "labelnative = " + str(labels_native) + ";\n")

        new_spm.write(design_outputs + "bias.warped = " + str(bias_warped) + ";\n")
        new_spm.write(design_outputs + "las.native = " + str(las_native) + ";\n")
        new_spm.write(design_outputs + "las.warped = " + str(las_warped) + ";\n")
        new_spm.write(design_outputs + "las.dartel = " + str(las_dartel) + ";\n")
        new_spm.write(design_outputs + "jacobianwarped = " + str(jacobian_warped) + ";\n")
        new_spm.write(design_outputs + "warps = " + output_warps + ";\n")

        new_spm.write("spm('defaults', 'PET');\n")
        new_spm.write("spm_jobman('initcfg');\n")
        new_spm.write("spm_jobman('run',matlabbatch);\n")
        new_spm.close()

        # os.system('%s run %s' % (self.spm_path, mfile))

    @staticmethod
    def create_mfile_model_pet_with_1_cov(mfile_name, save_dir, group1, group1_ages, group2, group2_ages,
                                          mask, dependence=0, variance=1, gmscaling=0, ancova=0, global_norm=1):

        design_type = "matlabbatch{1}.spm.stats.factorial_design."

        new_spm = open(mfile_name, "w")
        new_spm.write(
            design_type + "dir = {'" + save_dir + "/'};" + "\n" +
            "%%" + "\n" +
            design_type + "des.t2.scans1 = {" + "\n"
        )

        for image in group1:
            new_spm.write("'" + image + ",1'" + "\n")
        new_spm.write(
            "};" + "\n" +
            "%%" + "\n"
        )

        new_spm.write(design_type + "des.t2.scans2 = {" + "\n")

        for image in group2:
            new_spm.write("'" + image + ",1'" + "\n")
        new_spm.write("};" + "\n")

        new_spm.write(
            design_type + "des.t2.dept = " + str(dependence) + ";" + "\n" +
            design_type + "des.t2.variance = " + str(variance) + ";" + "\n" +
            design_type + "des.t2.gmsca = " + str(gmscaling) + ";" + "\n" +
            design_type + "des.t2.ancova = " + str(ancova) + ";" + "\n"
        )

        new_spm.write(design_type + "cov.c = [")
        for age in group1_ages:
            new_spm.write(str(age) + "\n")
        for age in group2_ages:
            new_spm.write(str(age) + "\n")
        new_spm.write("];" + "\n" + "%%" + "\n")

        new_spm.write(
            design_type + "cov.cname = 'Age';" + "\n" +
            design_type + "cov.iCFI = 1;" + "\n" +
            design_type + "cov.iCC = 5;" + "\n" +
            design_type + "multi_cov = struct('files', {}, 'iCFI', {}, 'iCC', {});" + "\n" +
            design_type + "masking.tm.tm_none = 1;" + "\n" +
            design_type + "masking.im = 0;" + "\n" +
            design_type + "masking.em = {'" + mask + ",1'};" + "\n" +
            design_type + "globalc.g_omit = 1;" + "\n" +
            design_type + "globalm.gmsca.gmsca_no = 1;" + "\n" +
            design_type + "globalm.glonorm = " + str(global_norm) + ";" + "\n"
        )

        new_spm.close()

    @staticmethod
    def create_mfile_model_pet_with_2_cov(mfile_name, save_dir, group1_dir, group1, group1_ages, group1_gluc,
                                          group2_dir, group2, group2_ages,
                                          group2_gluc, mask, dependence=0, variance=0, gmscaling=0, ancova=1,
                                          global_norm=0):

        design_type = "matlabbatch{1}.spm.stats.factorial_design."

        new_spm = open(mfile_name, "w")
        new_spm.write(
            design_type + "dir = {'" + save_dir + "/'};" + "\n" +
            "%%" + "\n" +
            design_type + "des.t2.scans1 = {" + "\n"
        )

        for image in group1:
            new_spm.write("'" + group1_dir + "/" + image + ",1'" + "\n")
        new_spm.write(
            "};" + "\n" +
            "%%" + "\n"
        )

        new_spm.write(design_type + "des.t2.scans2 = {" + "\n")

        for image in group2:
            new_spm.write("'" + group2_dir + "/" + image + ",1'" + "\n")
        new_spm.write("};" + "\n")

        new_spm.write(
            design_type + "des.t2.dept = " + str(dependence) + ";" + "\n" +
            design_type + "des.t2.variance = " + str(variance) + ";" + "\n" +
            design_type + "des.t2.gmsca = " + str(gmscaling) + ";" + "\n" +
            design_type + "des.t2.ancova = " + str(ancova) + ";" + "\n"
        )

        new_spm.write(design_type + "cov(1).c = [")
        for age in group1_ages:
            new_spm.write(str(age) + "\n")
        for age in group2_ages:
            new_spm.write(str(age) + "\n")
        new_spm.write("];" + "\n" + "%%" + "\n")

        new_spm.write(
            design_type + "cov(1).cname = 'Age';" + "\n" +
            design_type + "cov(1).iCFI = 1;" + "\n" +
            design_type + "cov(1).iCC = 5;" + "\n"
        )

        new_spm.write(design_type + "cov(2).c = [")
        for gluc in group1_gluc:
            new_spm.write(str(gluc) + "\n")
        for gluc in group2_gluc:
            new_spm.write(str(gluc) + "\n")
        new_spm.write("];" + "\n" + "%%" + "\n")

        new_spm.write(
            design_type + "cov(2).cname = 'Glucose';" + "\n" +
            design_type + "cov(2).iCFI = 1;" + "\n" +
            design_type + "cov(2).iCC = 5;" + "\n"
        )

        new_spm.write(
            design_type + "multi_cov = struct('files', {}, 'iCFI', {}, 'iCC', {});" + "\n" +
            design_type + "masking.tm.tm_none = 1;" + "\n" +
            design_type + "masking.im = 0;" + "\n" +
            design_type + "masking.em = {'" + mask + ",1'};" + "\n" +
            design_type + "globalc.g_omit = 1;" + "\n" +
            design_type + "globalm.gmsca.gmsca_no = 1;" + "\n" +
            design_type + "globalm.glonorm = " + str(global_norm) + ";" + "\n"
        )

        new_spm.close()

    @staticmethod
    def create_mfile_estimate_model(mfile_name, spm_mat):

        design_type = "matlabbatch{1}.spm.stats.fmri_est."

        new_spm = open(mfile_name, "w")
        new_spm.write(design_type + "spmmat = {'" + spm_mat + "'};\n")
        new_spm.write(design_type + "write_residuals = 0;")
        new_spm.write(design_type + "method.Classical = 1;")

        new_spm.close()

    @staticmethod
    def create_mfile_contrast(mfile_name, spm_mat, contrast_name='contrast', contrast='[1 -1 0]'):

        design_type = "matlabbatch{1}.spm.stats.con."

        new_spm = open(mfile_name, "w")

        new_spm.write(design_type + "spmmat = {'" + spm_mat + "'};\n")
        new_spm.write(design_type + "consess{1}.tcon.name = '" + contrast_name + "';\n")
        new_spm.write(design_type + "consess{1}.tcon.weights =" + contrast + ";\n")
        new_spm.write(design_type + "consess{1}.tcon.sessrep = 'none';\n")
        new_spm.write(design_type + "delete = 0;\n")

        new_spm.close()

    @staticmethod
    def spm_map_2_cohens_d(img, out, len_1, len_2):
        """
        Converts an image to cohens_d
        Expected input is a spmT_0001.nii file
        """
        img = nib.load(img)
        data = img.get_fdata()
        d_coeff = np.sqrt(1 / len_1 + 1 / len_2)
        data = data * d_coeff
        d_img = nib.AnalyzeImage(data, img.affine, img.header)
        nib.save(d_img, out)

    @staticmethod
    def get_tvalue_thresholds_FDR(img_, n1, n2):

        from scipy.stats import t
        # Load the NIFTI image
        img = nib.load(img_)

        # Get the data from the image
        data = img.get_fdata()

        # Flatten the data to get a 1D array of t-values
        t_values = data.flatten()
        t_values = abs(np.unique(t_values[t_values != 0]))
        t_values = np.sort(t_values)

        df = n1 + n2 - 2
        p_values = t.sf(abs(t_values), df=df)
        indx = np.where(p_values < 0.05)
        thresholded = t_values[indx]
        thres = np.percentile(thresholded, 5)

        d_coeff = np.sqrt(1 / n1 + 1 / n2)
        cohens_thres = thres * d_coeff

        print(cohens_thres)
        new_file_name = join(dirname(img_), 'cohensd_thres.txt')
        new_file = open(new_file_name, "w")
        new_file.write(str(cohens_thres))
        new_file.close()
