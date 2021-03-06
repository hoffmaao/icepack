# Copyright (C) 2019 by Daniel Shapero <shapero@uw.edu>
#
# This file is part of icepack.
#
# icepack is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# The full text of the license can be found in the file LICENSE in the
# icepack source directory or at <http://www.gnu.org/licenses/>.

r"""Routines for fetching the glaciological data sets used in the demos"""

import os
from getpass import getpass
import requests
import pooch

def _earthdata_downloader(url, output_file, dataset):
    username = os.environ.get('EARTHDATA_USERNAME')
    if username is None:
        username = input('EarthData username: ')

    password = os.environ.get('EARTHDATA_PASSWORD')
    if password is None:
        password = getpass('EarthData password: ')

    login = requests.get(url)
    downloader = pooch.HTTPDownloader(auth=(username, password))
    downloader(login.url, output_file, dataset)


nsidc_url = 'https://daacdata.apps.nsidc.org/pub/DATASETS/'

measures_antarctica = pooch.create(
    path=pooch.os_cache('icepack'),
    base_url=nsidc_url + 'nsidc0754_MEASURES_antarctic_ice_vel_phase_map_v01/',
    registry={
        'antarctic_ice_vel_phase_map_v01.nc':
        'fa0957618b8bd98099f4a419d7dc0e3a2c562d89e9791b4d0ed55e6017f52416'
    }
)

def fetch_measures_antarctica():
    return measures_antarctica.fetch('antarctic_ice_vel_phase_map_v01.nc',
                                     downloader=_earthdata_downloader)


bedmap2 = pooch.create(
    path=pooch.os_cache('icepack'),
    base_url='https://secure.antarctica.ac.uk/data/bedmap2/',
    registry={
        'bedmap2_tiff.zip':
        'f4bb27ce05197e9d29e4249d64a947b93aab264c3b4e6cbf49d6b339fb6c67fe'
    }
)

def fetch_bedmap2():
    filenames = bedmap2.fetch('bedmap2_tiff.zip', processor=pooch.Unzip())
    return [f for f in filenames if os.path.splitext(f)[1] == '.tif']


outlines_url = 'https://raw.githubusercontent.com/icepack/glacier-meshes/'
outlines_commit = '9306972327a127c4c4bdd3b5f61d2102307c2baa'
larsen_outline = pooch.create(
    path=pooch.os_cache('icepack'),
    base_url=outlines_url + outlines_commit + '/glaciers/',
    registry={
        'larsen.geojson':
        '74a632fcb7832df1c2f2d8c04302cfcdb3c1e86e027b8de5ba10e98d14d94856'
    }
)

def fetch_larsen_outline():
    return larsen_outline.fetch('larsen.geojson')


moa = pooch.create(
    path=pooch.os_cache('icepack'),
    base_url=nsidc_url + 'nsidc0593_moa2009/geotiff/',
    registry={
        'moa750_2009_hp1_v01.1.tif.gz':
        '90d1718ea0971795ec102482c47f308ba08ba2b88383facb9fe210877e80282c'
    }
)

def fetch_mosaic_of_antarctica():
    return moa.fetch('moa750_2009_hp1_v01.1.tif.gz',
                     downloader=_earthdata_downloader,
                     processor=pooch.Decompress())
