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

def main():
    all_scripts=re.findall('(?<=precise.scripts.)[a-z_]+', open('setup.py').read())
    package_scripts("precise-all", "precise", all_scripts, True)
    package_scripts("precise-engine", "precise-engine", ["engine"], False)

    tar_1 = join("dist",tar_name("precise-all"))
    tar_2 = join("dist",tar_name("precise-engine"))
    print("Wrote to {} and {}".format(tar_1, tar_2))

def package_scripts(tar_prefix, combined_folder, scripts, train_libs):
    """Use pyinstaller to create EXEs for the scripts
    and bundle them to a tar.gz file in dist/.

    Args:
        tar_prefix (str): The prefix of the output tar.gz file.
        combined_folder (str): The name of the directory in dist on which put all the files produced by pyinstaller.
    
    """
    completed_file=join("dist", "completed_{}.txt".format(combined_folder))
    if not os.path.isfile(completed_file):
        delete_dir_if_exists(join("dist", combined_folder))
    
    for script in scripts:
        exe = "precise-{}".format(script.replace('_', '-'))
        if filecontains(completed_file, exe):
            continue
        
        spec_path = createSpecFile(script, train_libs)
        if os.system("pyinstaller -y {} --workpath=build/{}".format(spec_path, exe)) != 0:
            raise Exception("pyinstaller error")

        if exe != combined_folder:
            dir_util.copy_tree(join("dist",exe), join("dist",combined_folder), update=1)
            shutil.rmtree(join("dist",exe))
            shutil.rmtree(join("build",exe))
        with open(completed_file, 'a') as f:
            f.write(exe + '\n')
        
    out_name = tar_name(tar_prefix)
    make_tarfile(join('dist', combined_folder), join('dist', out_name))
    with open(join('dist', "{}.md5".format(out_name)), 'w') as md5file:
        md5file.write(filemd5(join('dist', out_name)))
    
def filemd5(fname):
    """Calculate md5 hash of a file.

    Args:
        fname (str): (The path of) the file to be hashed.
    
    Returns:
        The md5 hash of the file in a hexadecimal string.
    
    """
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def make_tarfile(source_dir, output_filename):
    """Compress a directory into a gzipped tar file.

    Args:
        source_dir (str): (The path of) the directory to be compressed.
        output_filename (str): The tar.gz file name/path the directory should be compressed to.
    
    """
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=basename(source_dir))

def tar_name(tar_prefix):
    """Generate a name for a tar.gz file which includes the version, the os name (windows)
    and the architacture.

    Args:
        tar_prefix (str): The tar.gz filename will start with this value.
    
    Returns:
        A string in the following format: "{tar_prefix}_{version}_win_{archname}.tar.gz".
    
    """
    arch = platform.machine()
    return "{tar_prefix}_{version}_win_{archname}.tar.gz".format(
                                                            tar_prefix=tar_prefix,
                                                            version=__version__,
                                                            archname = arch)

def filecontains(path, string):
    """Check if a file contains a given phrase.
    
    Args:
        path (str): (The path of) the file to search in.
        string (str): The phrase to look for.
    
    Returns:
        True if the given file contains the given phrase, False otherwise.

    """
    try:
        f = open(path)
        content = f.read()
        f.close()
        return string in content
    except FileNotFoundError:
        return False

def delete_dir_if_exists(folder):
    """Delete a folder, ignore if it does not exist.

    Args:
        folder (str): The folder to be deleted.
    
    """

    try:
        shutil.rmtree(folder)
    except FileNotFoundError:
        pass

def createSpecFile(script, train_libs):
    """Create a pyinstaller spec file based on precise.template.spec.
    
    Args:
        script (str): the python script for which this spec file is intended.
        train_libs (bool): whether the spec should include the training libs.
    
    Returns:
        The path of the created spec file.
    
    """
    with tempfile.NamedTemporaryFile(mode = 'w',suffix='.spec', delete=False) as temp_file:
        spec_path = temp_file.name
        with open("precise.template.spec") as template_spec:
            spec = template_spec.read()                                \
                                .replace("%%SCRIPT%%",script)          \
                                .replace("%%TRAIN_LIBS%%", str(train_libs)) \
                                .replace("%%STRIP%%", "False")
        temp_file.write(spec + '\n')
    return spec_path

if __name__ == '__main__':
    main()
