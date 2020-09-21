# This file should not be used directly but using build.bat

import os
import os.path
from os.path import join, normpath, basename
import platform
import re
import tempfile
import shutil
import glob
import tarfile
from distutils import dir_util

from precise import __version__


import hashlib
def filemd5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=basename(source_dir))

def tar_name(tar_prefix):
    arch = platform.machine()
    return "{tar_prefix}_{version}_win_{archname}.tar.gz".format(
                                                            tar_prefix=tar_prefix,
                                                            version=__version__,
                                                            archname = arch)

def filecontains(path, string):
    try:
        f = open(path)
        content = f.read()
        f.close()
        return string in content
    except FileNotFoundError:
        return False

def package_scripts(tar_prefix, combined_folder, scripts, train_libs):
    completed_file=join("dist", "completed_{}.txt".format(combined_folder))
    if not os.path.isfile(completed_file):
        try:
            shutil.rmtree(join("dist",combined_folder))
        except FileNotFoundError:
            pass
    
    for script in scripts:
        exe = "precise-{}".format(script.replace('_', '-'))
        if filecontains(completed_file, exe):
            continue
        with tempfile.NamedTemporaryFile(mode = 'w',suffix='.spec', delete=False) as temp_file:
            temp_path = temp_file.name
            with open("precise.template.spec") as template_spec:
                spec = template_spec.read()                                \
                                    .replace("%%SCRIPT%%",script)          \
                                    .replace("%%TRAIN_LIBS%%", train_libs) \
                                    .replace("%%STRIP%%", "False")
            temp_file.write(spec + '\n')
        if os.system("pyinstaller -y {} --workpath=build/{}".format(temp_path, exe)) != 0:
            raise Exception("pyinstaller error")
        print(temp_path)
        if exe != combined_folder:
            dir_util.copy_tree(join("dist",exe), join("dist",combined_folder), update=1)
            shutil.rmtree(join("dist",exe))
            shutil.rmtree(join("build",exe))
        with open(completed_file, 'a') as f:
            f.write(exe + '\n')
        
    out_name = tar_name(tar_prefix)
    make_tarfile(join('dist', out_name), join('dist', combined_folder))
    with open(join('dist', "{}.md5".format(out_name)), 'w') as md5file:
        md5file.write(filemd5(join('dist', out_name)))
    
def main():
    all_scripts=re.findall('(?<=precise.scripts.)[a-z_]+', open('setup.py').read())
    package_scripts("precise-all", "precise", all_scripts, "True")
    package_scripts("precise-engine", "precise-engine", ["engine"], "False")

    tar_1 = join("dist",tar_name("precise-all"))
    tar_2 = join("dist",tar_name("precise-engine"))
    print("Wrote to {} and {}".format(tar_1, tar_2))

if __name__ == '__main__':
    main()
