"""
:Synopsis: script for GLM and stats on IBC datasets

:Author: THIRION Bertrand

"""

import os
import glob
from joblib import Parallel, delayed
from pypreprocess.conf_parser import _generate_preproc_pipeline
from ibc_public.data_utils import get_subject_session


def first_level(subject_dic, mask_img, compcorr=True,
                smooth=None):
    """ Run the first-level analysis (GLM fitting + statistical maps)
    in a given subject
    
    Parameters
    ----------
    subject_dic: dict,
                 exhaustive description of an individual acquisition
    additional_regressors: dict or None,
                 additional regressors provided as an already sampled 
                 design_matrix
                 dictionary keys are session_ids
    compcorr: Bool, optional,
              whetherconfound estimation and removal should be carried out or not
    smooth: float or None, optional,
            how much the data should spatially smoothed during masking
    """
    import nibabel as nib
    import numpy as np
    from nistats.design_matrix import make_design_matrix
    from nilearn.image import high_variance_confounds
    import pandas as pd
    from nistats.first_level_model import FirstLevelModel
    
    # experimental paradigm meta-params
    motion_names = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']
    hrf_model = 'spm'
    hfcut = subject_dic['hfcut']
    drift_model = subject_dic['drift_model']
    tr = subject_dic['TR']

    for session_id, fmri_path, onset, motion_path in zip(
            subject_dic['session_id'], subject_dic['func'],
            subject_dic['onset'], subject_dic['realignment_parameters']):
        n_scans = nib.load(fmri_path).shape[3]

        # motion parameters
        motion = np.loadtxt(motion_path)
        # define the time stamps for different images
        frametimes = np.linspace(0, (n_scans - 1) * tr, n_scans)
        
        confounds = motion
        confound_names = motion_names
        if compcorr:
            confounds = high_variance_confounds(fmri_path, mask_img=mask_img)
            confounds = np.hstack((confounds, motion))
            confound_names = ['conf_%d' % i for i in range(5)] + motion_names
                    
        paradigm = pd.read_csv(onset, sep='\t')
        trial_type = paradigm.trial_type.values
        audio_right_hands = ['audio_right_hand_%d' % i for i in range(5)]
        audio_left_hands = ['audio_left_hand_%d' % i for i in range(5)]
        video_right_hands = ['video_right_hand_%d' % i for i in range(5)]
        video_left_hands = ['video_left_hand_%d' % i for i in range(5)]
        trial_type[trial_type == 'audio_right_hand'] = audio_right_hands
        trial_type[trial_type == 'audio_left_hand'] = audio_left_hands
        trial_type[trial_type == 'video_right_hand'] = video_right_hands
        trial_type[trial_type == 'video_left_hand'] = video_left_hands
        
        # create the design matrix
        design_matrix = make_design_matrix(
            frametimes, paradigm, hrf_model=hrf_model, drift_model=drift_model,
            period_cut=hfcut, add_regs=confounds,
            add_reg_names=confound_names)
        
        # create the relevant contrasts
        names = design_matrix.columns
        n_regressors = len(names)
        interest = audio_right_hands + audio_left_hands + video_right_hands + video_left_hands
        con = dict([(names[i], np.eye(n_regressors)[i]) for i in range(n_regressors)])
        contrasts = dict([(contrast, con[contrast]) for contrast in interest])

        subject_session_output_dir = os.path.join(
            subject_dic['output_dir'], 'res_stats_%s' % session_id)

        if not os.path.exists(subject_session_output_dir):
            os.makedirs(subject_session_output_dir)
        design_matrix.to_csv(os.path.join(subject_session_output_dir, 'design_matrix.npz'))

        fmri_glm = FirstLevelModel(mask=mask_img, t_r=tr, slice_time_ref=.5,
                                   smoothing_fwhm=smooth).fit(
                                       fmri_path, design_matrices=design_matrix)
    
        # compute contrasts
        for contrast_id, contrast_val in contrasts.iteritems():
            print "\tcontrast id: %s" % contrast_id
        
            # store stat maps to disk
            for map_type in ['z_score']:
                stat_map = fmri_glm.compute_contrast(
                    contrast_val, output_type=map_type)
                map_dir = os.path.join(
                    subject_session_output_dir, '%s_maps' % map_type)
                if not os.path.exists(map_dir):
                    os.makedirs(map_dir)
                map_path = os.path.join(map_dir, '%s.nii.gz' % contrast_id)
                print "\t\tWriting %s ..." % map_path
                stat_map.to_filename(map_path)
            

def generate_glm_input(jobfile, smooth=None, lowres=False):
    """ retrun a list of dictionaries that represent the data available
    for GLM analysis"""
    list_subjects, params = _generate_preproc_pipeline(jobfile)
    output = []
    for subject in list_subjects:
        output_dir = subject.output_dir
        print(output_dir)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # use normalized anat
        anat = os.path.join(os.path.dirname(subject.anat), 'w' + os.path.basename(subject.anat))

        # use normalized fMRI
        basenames = ['wr' + os.path.basename(func_)[:-3] for func_ in subject.func]
        dirnames = [os.path.dirname(func_) for func_ in subject.func] 
        func = [os.path.join(dirname, basename + '.gz')
               for (dirname, basename) in zip(dirnames, basenames)]
        if lowres:
            func = [f.replace('derivatives', '3mm') for f in func]
        
        realignment_parameters = [
            os.path.join(dirname, 'rp_' + basename[2:-4] + '.txt')
            for (dirname, basename) in zip(dirnames, basenames)]

        # misc report directories
        reports_output_dir = os.path.join(output_dir, 'reports')
        report_log_filename = os.path.join(reports_output_dir, 'report_log.html')
        report_preproc_filename = os.path.join(
            reports_output_dir, 'report_preproc.html')
        report_filename = os.path.join(reports_output_dir, 'report.html')
        tmp_output_dir = os.path.join(output_dir, 'tmp')        
        subject_ = {
            'scratch': output_dir,
            'output_dir':output_dir,
            'session_output_dirs': subject.session_output_dirs,
            'anat_output_dir': subject.anat_output_dir,
            'tmp_output_dir': tmp_output_dir,
            'data_dir': subject.data_dir,
            'subject_id': subject.subject_id,
            'session_id':subject.session_id,
            'TR': subject.TR,
            'drift_model':subject.drift_model,
            'hfcut': subject.hfcut,
            'time_units': subject.time_units,
            'hrf_model': subject.hrf_model,
            'anat': anat,
            'onset': subject.onset,
            'report': True,
            'reports_output_dir': reports_output_dir,
            'report_log_filename': report_log_filename,
            'report_preproc_filename': report_preproc_filename,
            'report_filename': report_filename,
            'basenames': basenames,
            'func': func,
            'n_sessions': len(func),
            'realignment_parameters': realignment_parameters,
        }
        output.append(subject_)
    return output


def _adapt_jobfile(jobfile, subject, output_name, session=None):
    """ small utility to create temporary jobfile"""
    f1 = open(jobfile, 'r')
    f2 = open(output_name, 'w')
    for line in f1.readlines():
        if session is None:
            f2.write(line.replace('sub-01', subject))
        else:
            f2.write(line.replace('sub-01', subject).replace('ses-*', session))

    f1.close()
    f2.close()
    return output_name


def run_subject_glm(jobfile, protocol, subject, session=None, smooth=None, lowres=False):
    """ Create jobfile and run it """
    mask_img = 'gm_mask_3mm.nii.gz'
    output_name = os.path.join(
        '/tmp', os.path.basename(jobfile)[:-4] + '_%s.ini' % subject)
    _adapt_jobfile(jobfile, subject, output_name, session)
    list_subjects_update = generate_glm_input(output_name, smooth, lowres)
    # list_subjects_update = generate_glm_input(jobfile, smooth, lowres)
    for subject in list_subjects_update:
        subject['onset'] = [onset for onset in subject['onset'] if onset is not None]
        # clean_subject(subject)
        if len(subject['session_id']) > 0:
            first_level(subject, compcorr=True, smooth=smooth, mask_img=mask_img)
            # fixed_effects_analysis(subject, mask_img=mask_img)


def resample(img, reference, target=None):
    from nilearn.image import resample_to_img
    rimg = resample_to_img(img, reference)
    if target is not None:
        rimg.to_filename(target)
    else:
        rimg.to_filename(img)
    print(img)


if __name__ == '__main__':
    smooth = 5
    for protocol in ['archi', 'screening']: # 'screening', 
        jobfile = 'ibc_preproc_screening.ini'
        subject_session = get_subject_session(protocol)
        Parallel(n_jobs=4)(
            delayed(run_subject_glm)(jobfile, protocol, subject, session, smooth, lowres=True)
            for (subject, session) in subject_session)

    import numpy as np
    from nilearn.input_data import NiftiMasker
    from sklearn.svm import SVC
    from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
    mask_img = 'gm_mask_3mm.nii.gz'
    masker = NiftiMasker(mask_img=mask_img, memory="nilearn_cache", memory_level=1)
    cv = LeaveOneGroupOut()
    subjects = np.unique([x[0] for x in subject_session])
    for subject in subjects:
        imgs = glob.glob(os.path.join(
            '/neurospin/tmp/archi_motor/derivatives', subject,
            'ses-*/res_stats_archi_standard_*/z_score_maps/*.nii.gz'))
        y = np.array(['right' in img for img in imgs])
        _, session = np.unique([path.split('ses')[1].split('z_score_map')[0]
                                for path in imgs], return_inverse=True)
        X = masker.fit_transform(imgs)
        svc = SVC(kernel='linear')
        cv_scores = cross_val_score(svc, X, y, cv=cv, groups=session)
        print(cv_scores.mean())
