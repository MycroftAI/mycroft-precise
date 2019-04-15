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
    local completed_file="dist/completed_$combined_folder.txt"

    if ! [ -f "$completed_file" ]; then
        rm -rf "dist/$combined_folder"
    fi
    mkdir -p "dist/$combined_folder"

    for script in $scripts; do
        exe=precise-$(echo "$script" | tr '_' '-')
        if [ -f "$completed_file" ] && grep -qF "$exe" "$completed_file"; then
            continue
        fi
        tmp_name=$(mktemp).spec
        cat "precise.template.spec" | replace "%%SCRIPT%%" "$script" | replace "%%TRAIN_LIBS%%" "$train_libs" > "$tmp_name"
        pyinstaller -y "$tmp_name"
        if [ "$exe" != "$combined_folder" ]; then
            cp -R dist/$exe/* "dist/$combined_folder"
            rm -rf "dist/$exe" "build/$exe"
        fi
        echo "$exe" >> "$completed_file"
    done

    out_name=$(tar_name "$tar_prefix")
    cd dist
    tar czvf "$out_name" "$combined_folder"
    md5sum "$out_name" > "$out_name.md5"
    cd ..
}

set -eE

./setup.sh
source .venv/bin/activate
pip install pyinstaller

all_scripts=$(grep -oP '(?<=precise.scripts.)[a-z_]+' setup.py)
package_scripts "precise-all" "precise" "$all_scripts" True
package_scripts "precise-engine" "precise-engine" "engine" False

tar_1=dist/$(tar_name precise-all)
tar_2=dist/$(tar_name precise-engine)
echo "Wrote to $tar_1 and $tar_2"
