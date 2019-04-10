#!/usr/bin/env bash


tar_name() {
    local tar_prefix=$1
    echo "${tar_prefix}_$(precise-engine --version 2>&1)_$(uname -m).tar.gz"
}

replace() {
    local pattern=$1
    local replacement=$2
    sed -e "s/$pattern/$replacement/gm"
}

package_scripts() {
    local tar_prefix=$1
    local combined_folder=$2
    local scripts=$3
    local train_libs=$4
    
    rm -rf dist/

    for script in $scripts; do
        tmp_name=$(mktemp).spec
        cat "precise.template.spec" | replace "%%SCRIPT%%" "$script" | replace "%%TRAIN_LIBS%%" "$train_libs" > "$tmp_name"
        pyinstaller -y "$tmp_name"
    done

    local items=dist/*
    for i in $items; do
        mkdir -p "dist/$combined_folder"
        if [ "$(readlink -f "$i")" != "$(readlink -f "dist/$combined_folder")" ]; then
            cp -R $i/* "dist/$combined_folder"
        fi
    done

    out_name=$(tar_name "$tar_prefix")
    cd dist
    tar czvf "$out_name" "$tar_prefix"
    cd ..
    mv "dist/$out_name" .
}

set -eE

./setup.sh
source .venv/bin/activate
pip install pyinstaller

all_scripts=$(grep -oP '(?<=precise.scripts.)[a-z_]+' setup.py)
package_scripts "precise-all" "precise" "$all_scripts" True
package_scripts "precise-engine" "precise-engine" "engine" False

tar_1=$(tar_name precise-all)
tar_2=$(tar_name precise-engine)
echo "Wrote to $tar_1 and $tar_2"
