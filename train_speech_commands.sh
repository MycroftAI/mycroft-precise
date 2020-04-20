#!/usr/bin/env bash

if [ "$1" != "--clean" ] && [ "$1" != "" ]; then
    echo "Usage: $0 [--clean]"
    exit 1
fi

cd "$(dirname "$0")"  # Cd to script location
set -eE

dataset_url="http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
dataset_md5="6b74f3901214cb2c2934e98196829835"
dataset_filename="speech_commands_v0.02.tar.gz"
dataset_folder="speech_commands"
demo_class="marvin"

if [ "$1" = "--clean" ]; then
    echo "Cleaning old data..."
    rm "data/$dataset_filename"
    rm -r "data/$dataset_folder"
fi

echo "Downloading speech commands dataset..."
cur_md5=$(md5sum "data/$dataset_filename" | awk '{print $1}')
if [ "$cur_md5" != "$dataset_md5" ] || [ ! -d "data/$dataset_folder" ]; then
    mkdir -p "data/$dataset_folder"
    if [ "$cur_md5" != "$dataset_md5" ]; then
        wget "$dataset_url" -O "data/$dataset_filename"
    fi
    pushd "data/$dataset_folder"
    tar xvf "../$dataset_filename"
    popd 2>/dev/null
else
    echo "Skipping, already downloaded."
fi

pushd "data/$dataset_folder"
classes=$(find * -maxdepth 0 -type d -name '[a-zA-Z]*')
for class in $classes; do
    if [ -f "tags-$class.txt" ]; then
        echo "Already generated tags-$class.txt."
        continue
    fi
    echo "Generating tags-$class.txt..."
    {
        find "$class" -name '*.wav' | {
            while read line; do
                printf "${line%.wav}\twake-word\n"
            done
        }
        for other_class in $classes; do
            if [ "$class" = "$other_class" ]; then
                continue
            fi
            find "$other_class" -name '*.wav' | {
                while read line; do
                    printf "${line%.wav}\tnot-wake-word\n"
                done
            }
        done
    } > "tags-$class.txt"
done
popd 2>/dev/null

echo "Setting up packages..."
./setup.sh
source .venv/bin/activate

echo "Dataset import complete."

mkdir -p models/
train_command="precise-train models/$demo_class.net data/$dataset_folder --tags-file data/$dataset_folder/tags-$demo_class.txt -e 3 --batch-size 128 --sensitivity 0.5"
echo ""
echo "Training $demo_class model with command:"
echo "$ $train_command"
echo ""
echo "Press any key to begin model training..."
read -n 1
eval "$train_command"

echo "Model saved to models/$demo_class.net"
echo ""

test_command="precise-test models/$demo_class.net data/$dataset_folder --tags-file data/$dataset_folder/tags-$demo_class.txt"
echo "Testing $demo_class model with command:"
echo "$ $test_command"
echo ""
echo "Press any key to test model..."
read -n 1
eval "$test_command"

listen_command="precise-listen models/$demo_class.net --sensitivity 0.5"
echo "Running $demo_class model against microphone with command:"
echo "$ $listen_command"
echo ""
echo "Note: This will continuously listen to the microphone and should activate with the word \"$demo_class\". When you are done testing you can exit with Ctrl+C."
echo "Press any key to test against microphone..."
read -n 1
eval "$listen_command"
