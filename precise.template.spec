# -*- mode: python -*-
block_cipher = None

from glob import iglob
from os.path import basename, dirname, abspath

script_name = '%%SCRIPT%%'
train_libs = %%TRAIN_LIBS%%
strip = True
site_packages = '.venv/lib/python3.6/site-packages/'

if train_libs:
    binaries = [
        (abspath(i), dirname(i.replace(site_packages, '')))
        for i in iglob(site_packages + "tensorflow/**/*.so", recursive=True)
    ]
else:
    binaries = []

a = Analysis(
    [abspath('precise/scripts/{}.py'.format(script_name))],
    pathex=['.'],
    binaries=binaries,
    datas=[],
    hiddenimports=['prettyparse', 'speechpy'],
    hookspath=[],
    runtime_hooks=[],
    excludes=['PySide', 'PyQt4', 'PyQt5', 'matplotlib'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher
)

for i in range(len(a.binaries)):
    dest, origin, kind = a.binaries[i]
    if '_pywrap_tensorflow_internal' in dest:
        a.binaries[i] = ('tensorflow.python.' + dest, origin, kind)

pyz = PYZ(
    a.pure, a.zipped_data,
    cipher=block_cipher
)

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name='precise-{}'.format(script_name.replace('_', '-')),
    debug=False,
    strip=strip,
    upx=True,
    console=True
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=strip,
    upx=True,
    name='precise-{}'.format(script_name.replace('_', '-'))
)
