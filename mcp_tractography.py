#! /usr/bin/env python

import os
import shutil
import shlex
import argparse
import json
import inspect

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from multiprocessing import Pool
from os.path import abspath, join, isfile
from subprocess import Popen, PIPE
from itertools import product
from nibabel import streamlines
from nibabel.streamlines.array_sequence import ArraySequence
from dipy.tracking.streamline import length, set_number_of_points


def run(cmd, live_verbose=False):
    print('\n' + cmd)
    p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE)
    output, error = p.communicate()
    if output:
        print(output.decode('latin_1'))
    if error:
        print(error.decode('latin_1'))


def assert_dir(dir_path):
    full_path = abspath(dir_path)
    if not os.path.isdir(full_path):
        print('Creating %s' % full_path)
        os.makedirs(full_path)


def remake_dir(dir_path):
    full_path = abspath(dir_path)
    if os.path.isdir(full_path):
        shutil.rmtree(full_path)
    os.makedirs(full_path)


def move(src, dest):
    print('Moving %s to %s' % (src, dest))
    shutil.move(src, dest)


def copy(src, dest):
    print('Copying %s to %s' % (src, dest))
    shutil.copyfile(src, dest)


def mrconvert(src, dest):
    cmd = 'mrconvert %s %s' % (src, dest)
    run(cmd)


def proc(subject, mrtrix_dir, n_threads=0):

    print('Performing processing of DWI data')

    assert_dir(mrtrix_dir)

    fa = join(mrtrix_dir, subject, 'fa.mif')
    if not isfile(fa):

        dwi = join(mrtrix_dir, subject, 'dwi_raw.nii.gz')
        bvecs = join(mrtrix_dir, subject, 'bvecs')
        bvals = join(mrtrix_dir, subject, 'bvals')
        json_file = join(mrtrix_dir, subject, 'info.json')

        # Convert data to mrtrix format
        raw = join(mrtrix_dir, subject, 'dwi_raw.mif')
        if not isfile(raw):
            cmd = 'mrconvert -fslgrad %s %s -json_import %s %s %s -force' % (
                    bvecs, bvals, json_file, dwi, raw)
            run(cmd)

        # Perform eddy correction
        prep = join(mrtrix_dir, subject, 'dwi_preproc.mif')
        if not isfile(prep):
            with open(json_file, 'r') as f:
                json_content = json.load(f)
            pe_dir = json_content['PhaseEncodingDirection']
            cmd = 'dwifslpreproc %s %s -rpe_none -pe_dir %s ' \
                  '-nthreads %i -force' % (
                   raw, prep, pe_dir, n_threads)
            run(cmd)

        # Create mask
        mask = join(mrtrix_dir, subject, 'mask.mif')
        if not isfile(mask):
            cmd = 'dwi2mask %s %s -force' % (prep, mask)
            run(cmd)

        # Compute tensor
        tensor = join(mrtrix_dir, subject, 'tensor.mif')
        if not isfile(tensor):
            b0 = join(mrtrix_dir, subject, 'b0.mif')
            mask = join(mrtrix_dir, subject, 'mask.mif')
            dwi = join(mrtrix_dir, subject, 'dwi_preproc.mif')
            cmd = 'dwi2tensor -b0 %s -mask %s %s %s -force' % (
                    b0, mask, dwi, tensor)
            run(cmd)

        # Compute metrics
        vector = join(mrtrix_dir, subject, 'vector.mif')
        adc = join(mrtrix_dir, subject, 'adc.mif')
        fa = join(mrtrix_dir, subject, 'fa.mif')
        ad = join(mrtrix_dir, subject, 'ad.mif')
        rd = join(mrtrix_dir, subject, 'rd.mif')

        if np.any([not isfile(f) for f in [vector, adc, fa, ad, rd]]):
            cmd = 'tensor2metric -vector %s -adc %s -fa %s -ad %s -rd %s \
                   -mask %s %s -force' % (
                   vector, adc, fa, ad, rd, mask, tensor)
            run(cmd)

    else:
        print('Subejct %s already processed. Skipping.' % subject)


def transform(subject, mrtrix_dir, template_dir):

    print('Computing transform from diffusion to template space')

    transform_dir = join(mrtrix_dir, subject, 'transform')
    assert_dir(transform_dir)

    fa_template = join(template_dir, 'fa_template_0.5mm.mif')
    template_mask = join(template_dir, 'template_mask_0.5mm.mif')
    warp1 = join(transform_dir, 'warp1.mif')

    if not os.path.isfile(warp1):
        warp2 = join(transform_dir, 'warp2.mif')
        fa = join(mrtrix_dir, subject, 'fa.mif')
        mask = join(mrtrix_dir, subject, 'mask.mif')
        trans = join(transform_dir, 'fa_template.mif')
        cmd = 'mrregister %s %s \
                -nl_warp %s %s \
                -mask1 %s -mask2 %s \
                -transformed %s \
                -type rigid_affine_nonlinear \
                -rigid_scale 0.25,0.5,0.8,1.0 \
                -affine_scale 0.7,0.8,1.0,1.0 \
                -nl_scale 0.5,0.75,1.0,1.0,1.0 \
                -nl_niter 5,5,5,5,5 \
                -datatype float32 -noreorientation -force' % (
                fa, fa_template, warp1, warp2, mask, template_mask, trans)
        run(cmd)
    else:
        print('Subject %s already processed. Skipping.' % subject)


def tracks(subject, mrtrix_dir, template_dir):

    print('Extracting MCP tracks')

    mcp_dir = join(template_dir, 'mcp')
    transform_dir = join(mrtrix_dir, subject, 'transform')
    tracks_dir = join(mrtrix_dir, subject, 'tracks')
    assert_dir(tracks_dir)

    # Create whole brain tractography
    mask = join(mrtrix_dir, subject, 'mask.mif')
    vector = join(mrtrix_dir, subject, 'vector.mif')
    tck = join(tracks_dir, 'tracks.tck')
    if not isfile(tck):
        cmd = 'tckgen -algorithm FACT -seed_image %s -mask %s -angle 30 \
                -select 100000 -minlength 20 %s %s -force' % (
                mask, mask, vector, tck)
        run(cmd)

    # Transfer tracks to template space
    warp = join(transform_dir, 'warp2.mif')
    tck_in = join(tracks_dir, 'tracks.tck')
    tck_temp = join(tracks_dir, 'tracks.template.tck')
    if not isfile(tck_temp):
        cmd = 'tcktransform %s %s %s -force' % (tck_in, warp, tck_temp)
        run(cmd)

    # Select tracks
    for hemi in ['L', 'R']:
        tck_out = join(tracks_dir, 'MCP_%s.template.tck' % hemi)
        roi = join(mcp_dir, 'MCP_%s.nii.gz' % hemi)
        excl = join(mcp_dir, 'MCP_%s_exclude.nii.gz' % hemi)
        cmd = 'tckedit -minlength 15 -include %s -exclude %s %s %s -force' % (
                    roi, excl, tck_temp, tck_out)
        run(cmd)

    os.remove(tck_temp)


def trim_track(stl, target_len='floor', pts_dist=None, n_pts=None):

    seg_len = np.linalg.norm(np.diff(stl, axis=0), axis=1)
    seg_cumlen = np.cumsum(seg_len)

    if target_len == 'floor':
        lim = np.floor(seg_cumlen[-1])
    else:
        lim = target_len

    len_thr = [x < lim for x in seg_cumlen]
    if not np.any(len_thr):
        current = np.linalg.norm(stl[1, :]-stl[0])
        target = lim
        stretch = target/current
        stl_trim = np.vstack((stl[0, :].copy(),
                             stl[0, :] + (stl[1, :] - stl[0, :])*stretch))
    else:
        i = np.where(len_thr)[0][-1] + 1
        current = np.linalg.norm(stl[i+1, :]-stl[i])
        target = lim - seg_cumlen[i-1]
        stretch = target/current
        stl_trim = np.vstack((stl[:i+1, :].copy(),
                             stl[i, :] + (stl[i+1, :] - stl[i, :])*stretch))

    if pts_dist is not None and int(lim/pts_dist) + 1 >= 2:
        stl_trim = set_number_of_points(stl_trim, int(lim/pts_dist) + 1)
    else:
        raise ValueError('Steamline is shorter than requested pts_dist.')

    if n_pts is not None:
        stl_trim = set_number_of_points(stl_trim, n_pts)

    return stl_trim


def tcktrim(fin, fout, target_len='floor', pts_dist=None, n_pts=None):

    tck_in = streamlines.load(fin)
    stl = tck_in.streamlines.copy()
    stl_trim = []
    for stl_ in stl:
        stl_trim.append(trim_track(stl_, target_len=target_len,
                        pts_dist=pts_dist, n_pts=n_pts))
    tck_out = streamlines.tck.TckFile(streamlines.Tractogram(
                    ArraySequence(stl_trim),
                    affine_to_rasmm=tck_in.tractogram.affine_to_rasmm))
    tck_out.save(fout)


def filter(subject, mrtrix_dir, template_dir):

    print('Filtering tracks prior to tractometry')

    mcp_dir = join(template_dir, 'mcp')
    tracks_dir = join(mrtrix_dir, subject, 'tracks')

    for hemi in ['L', 'R']:

        # Limit anterior part of tracks to selection ROI
        tck_out = join(tracks_dir, 'MCP_%s.template.ant_lim.tck' % hemi)
        if not isfile(tck_out):
            tck_in = join(tracks_dir, 'MCP_%s.template.tck' % hemi)
            mask = join(mcp_dir, 'MCP_%s_ant_lim.nii.gz' % hemi)
            cmd = 'tckedit -force -mask %s %s %s -force' % (
                        mask, tck_in, tck_out)
            run(cmd)

        tck_out = join(tracks_dir, 'MCP_%s.template.filter.tck' % hemi)
        if not isfile(tck_out):
            fname = join(tracks_dir, 'MCP_%s.template.ant_lim.tck' % hemi)
            tck_in = streamlines.load(fname)
            stl = tck_in.streamlines.copy()

            # Discard everything below 15mm and above 45mm
            stl_len = length(stl).reshape([-1, 1])
            ind = [x > 15. and x < 45. for x in stl_len]
            ind = [x[0] for x in ind]
            stl_thr = stl[ind].copy()
            stl_len = length(stl_thr).reshape([-1, 1])

            plt.hist(stl_len, bins=100, density=True)
            plt.title(subject)
            plt.xlabel('Length (mm)')
            plt.ylabel('Count')
            fname = join(tracks_dir, 'MCP_%s.template.histo.png' % hemi)
            plt.savefig(fname, format='png')
            plt.clf()

            # Match starts and ends

            # Get all start and stop num_points
            starts = np.vstack([stl_[0, :] for stl_ in stl_thr])
            stops = np.vstack([stl_[-1, :] for stl_ in stl_thr])

            # Reverse streamline if stop is more anterior than start
            reverse = [start < stop
                       for start, stop in zip(starts[:, 1], stops[:, 1])]
            stl_flip = list(stl_thr.copy())
            for ns in np.where(reverse)[0]:
                stl_flip[ns] = stl_flip[ns][::-1, :]

            # Trim streamlines to 1mm segments
            stl_len = length(stl_flip).reshape([-1, 1])
            stl_trim = []
            for stl_ in stl_flip:
                stl_trim.append(trim_track(stl_, target_len='floor', pts_dist=1.))

            # Visualize mean start and stop of streamlines for QA
            pts = np.vstack(stl_trim)
            starts = np.vstack([stl_[0, :] for stl_ in stl_trim])
            stops = np.vstack([stl_[-1, :] for stl_ in stl_trim])
            plt.plot(pts[:, 0], pts[:, 1], 'b.')
            plt.plot(starts[:, 0], starts[:, 1], 'g.')
            plt.plot(stops[:, 0], stops[:, 1], 'r.')
            plt.title(subject)
            plt.tight_layout()
            fname = join(tracks_dir, 'MCP_%s.filter.png' % hemi)
            plt.savefig(fname, format='png')
            plt.clf()

            # Save filtered tracks
            stl_final = ArraySequence(stl_trim)
            tck_filter = streamlines.tck.TckFile(
                            streamlines.Tractogram(stl_final,
                            affine_to_rasmm=tck_in.tractogram.affine_to_rasmm)
                      )
            tck_filter.save(tck_out)

        # Transfer tracks back to diffusion space
        tck_out = join(tracks_dir, 'MCP_%s.filter.tck' % hemi)
        if not isfile(tck_out):
            warp = join(mrtrix_dir, subject, 'transform', 'warp1.mif')
            tck_in = join(tracks_dir, 'MCP_%s.template.filter.tck' % hemi)
            cmd = 'tcktransform %s %s %s -force' % (tck_in, warp, tck_out)
            run(cmd)


def sample_tractogram_data(subject, mrtrix_dir):

    metrics = ['fa', 'adc', 'ad', 'rd']

    tracks_dir = join(mrtrix_dir, subject, 'tracks')
    metrics_dir = join(mrtrix_dir, subject, 'metrics')
    assert_dir(metrics_dir)

    for hemi in ['L', 'R']:

        # Sample metrics
        for metric in metrics:
            tck = join(tracks_dir, 'MCP_%s.filter.tck' % hemi)
            data = join(mrtrix_dir, subject, '%s.mif' % metric)
            outfile = join(metrics_dir, 'MCP_%s_%s.dat' %
                           (hemi, metric))

            cmd = 'tcksample %s %s %s -force' % (tck, data, outfile)
            run(cmd)


def extend_streamline_data(data, target_len=45):
    data_ext = []
    for nd, data_ in enumerate(data):
        if len(data_) < target_len:
            data_ext.append(np.hstack((data_,
                            [np.nan]*(target_len-len(data_)))))
        else:
            data_ext.append(data_[:target_len])
    return np.vstack(data_ext)


def export_tractogram_data(subjects, mrtrix_dir, stats_dir, target_len=30):

    tract_dir = join(stats_dir, 'tractogram_data')
    assert_dir(tract_dir)

    metrics = ['fa', 'adc', 'ad', 'rd']
    hemilist = ['L', 'R']

    length = range(0, target_len + 1)
    df = pd.DataFrame(index=length)
    df.index.name = 'length'

    # Extract metrics
    for subject in subjects:

        print('Processing %s' % subject)

        for metric in metrics:

            df_ = pd.DataFrame(index=length)
            df_.index.name = 'length'

            for hemi in hemilist:

                metrics_dir = join(mrtrix_dir, subject, 'metrics')

                fname = join(metrics_dir, 'MCP_%s_%s.dat' % (hemi, metric))
                with open(fname, 'r') as f:
                    lines = f.readlines()
                data = np.array([[float(x) for x in line.strip().split()]
                                for line in lines[1:]])
                data = extend_streamline_data(data, target_len + 1)

                if metric != 'fa':
                    data = data * 1000

                # All tracts
                df_ = pd.DataFrame(data.T)
                fname = join(tract_dir, '%s.%s.%s.csv' % (
                             subject, hemi, metric))
                df_.to_csv(fname)

                # Median data
                col = '%s.%s.%s' % (subject, hemi, metric)
                df.loc[:, col] = np.nanmedian(data, axis=0)

    # Save data
    fname = join(tract_dir, 'median.tracks.csv')
    df.to_csv(fname)


def tractogram_trimmed_metrics(subjects, mrtrix_dir, stats_dir, target_len=5):

    # Trim streamlines to length of interest
    print('Trimming steamlines')
    
    hemilist = ['L', 'R']
    for subject in subjects:
        # subject = 'vco1014.test'

        tracks_dir = join(mrtrix_dir, subject, 'tracks')

        for hemi in hemilist:
            tck_final = join(tracks_dir, 'MCP_%s.filter.%imm.tck' % (
                            hemi, target_len))
            if not isfile(tck_final):
                tck_in = join(tracks_dir, 'MCP_%s.template.filter.tck' % hemi)
                tck_out = join(tracks_dir, 'MCP_%s.template.filter.%imm.tck' % (
                               hemi, target_len))
                tcktrim(tck_in, tck_out, target_len=target_len, pts_dist=1.)
                # Transfer tracks back to MNI space
                warp = join(mrtrix_dir, subject, 'transform', 'warp1.mif')
                cmd = 'tcktransform %s %s %s -force' % (
                        tck_out, warp, tck_final)
                run(cmd)

    print('Extracting metrics')

    metrics = ['fa', 'adc', 'ad', 'rd']
    cols = ['MCP.%s.%s' % (hemi, metric)
            for hemi, metric in product(hemilist, metrics)]
    df = pd.DataFrame(index=subjects, columns=cols)
    df.index.name = 'subjects'

    for metric in metrics:
        for hemi in hemilist:
            for subject in subjects:

                tracks_dir = join(mrtrix_dir, subject, 'tracks')
                metrics_dir = join(mrtrix_dir, subject, 'metrics')

                tck = join(tracks_dir, 'MCP_%s.filter.%imm.tck' % (
                        hemi, target_len))
                data = join(mrtrix_dir, subject, '%s.mif' % metric)
                outfile = join(metrics_dir, 'MCP_%s_%s.%imm.dat' %
                               (hemi, metric, target_len))
                if not isfile(outfile):
                    cmd = 'tcksample %s %s %s -force' % (tck, data, outfile)
                    run(cmd)

                with open(outfile, 'r') as f:
                    lines = f.readlines()
                data = np.array([[float(x) for x in line.strip().split()]
                                 for line in lines[1:]])
                if metric != 'fa':
                    data = data * 1000

                col = 'MCP.%s.%s' % (hemi, metric)
                df.loc[subject, col] = np.mean(np.median(
                                               np.vstack(data), axis=0))

    assert_dir(stats_dir)
    fname = join(stats_dir, 'tracts.mean.%imm.csv' % target_len)
    df.to_csv(fname)


if __name__ == '__main__':

    main_dir = abspath(join(inspect.getfile(inspect.currentframe()),
                            os.pardir))

    parser = argparse.ArgumentParser(description="MCP tractograpy using MRtrix3")
    mutex = parser.add_mutually_exclusive_group()
    mutex.add_argument("-s", "--subject", type=str, default=None,
                       help="Subjects to be processed")
    mutex.add_argument("-sl", "--subjects_list", type=str, default=None,
                       help="List of all subject to be used for training.")
    parser.add_argument("-i", "--import_dicoms", action="store_true",
                        help="Import DWI data from DICOMs.")
    parser.add_argument("-p", "--proc", action="store_true",
                        help="Perform processing of DWI data.")
    parser.add_argument("-tr", "--transform", action="store_true",
                        help="Compute transformation from MNI to diffusion space")
    parser.add_argument("-t", "--tracks", action="store_true",
                        help="Extract MCP tracts from whole-brain \
                              tractography.")
    parser.add_argument("-f", "--filter", action="store_true",
                        help="Filter tracks to perform tractogram analysis.")
    parser.add_argument("-std", "--sample_tractogram_data", action="store_true",
                        help="Extract data for tractogram along MCP tracks.")
    parser.add_argument("-etd", "--export_tractogram_data", action="store_true",
                        help="Export median tractogram data to CSV.")
    parser.add_argument("-ttm", "--tractogram_trimmed_metrics",
                        action="store_true",
                        help="Extract mean of data along tracts trimmed at a given length.")
    parser.add_argument("-tl", "--target_length", type=float, default=10,
                        help="Target length for trimming.")
    parser.add_argument("-n_threads", "--n_threads", type=int, default=0,
                        help="Number of thread to be used by dwipreproc.")
    parser.add_argument("-n_jobs", "--n_jobs", type=int, default=1,
                        help="Number of parallel jobs. Default 1.")
    args = parser.parse_args()

    # Post process arguments
    if args.subject:
        subjects = [args.subject]

    if args.subjects_list:
        with open(args.subjects_list, 'r') as f:
            subjects = [subject.strip() for subject in f.readlines()]

    # Define main directories
    dicoms_dir = join(main_dir, 'dicoms')
    mrtrix_dir = join(main_dir, 'mrtrix')
    template_dir = join(main_dir, 'population_template')
    stats_dir = join(main_dir, 'stats')

    # Run the commands

    pool = Pool(processes=args.n_jobs)

    if args.proc:
        params = list(product(subjects, [mrtrix_dir], [args.n_threads]))
        r = pool.starmap_async(proc, params)
        r.wait()

    if args.transform:
        params = list(product(subjects, [mrtrix_dir], [template_dir]))
        r = pool.starmap_async(transform, params)
        r.wait()

    if args.tracks:
        params = list(product(subjects, [mrtrix_dir], [template_dir]))
        r = pool.starmap_async(tracks, params)
        r.wait()

    if args.filter:
        params = list(product(subjects, [mrtrix_dir], [template_dir]))
        r = pool.starmap_async(filter, params)
        r.wait()

    if args.sample_tractogram_data:
        params = list(product(subjects, [mrtrix_dir]))
        r = pool.starmap_async(sample_tractogram_data, params)
        r.wait()

    pool.close()
    pool.join()

    if args.export_tractogram_data:
        export_tractogram_data(subjects, mrtrix_dir, stats_dir)

    if args.tractogram_trimmed_metrics:
        tractogram_trimmed_metrics(
                        subjects,
                        mrtrix_dir,
                        stats_dir,
                        args.target_length
        )
