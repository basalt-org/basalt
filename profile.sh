function profile() {
    local mojo_file=$1
    local mojo_name="${mojo_file%.mojo}"
    if [ ! -d "./temp" ]; then
        mkdir ./temp
    fi
    mojo build -O0 -I . $mojo_file
    mv "$mojo_name" "./run.exe"
    ~/perf record -F 99 -a -g -o ./temp/out.perf -- ./run.exe
    ~/perf script -i ./temp/out.perf | ~/FlameGraph/stackcollapse-perf.pl | ~/FlameGraph/flamegraph.pl > flamegraph.svg
    explorer.exe flamegraph.svg
    rm -rf ./temp
    rm -rf ./run.exe
}

profile "$1" 

# If cat /proc/sys/kernel/kptr_restrict = 1
# Run echo 0 | sudo tee /proc/sys/kernel/kptr_restrict
# Then sudo sysctl -p