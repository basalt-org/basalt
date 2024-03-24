function profile() {
    local mojo_file=$1
    local mojo_name="${mojo_file%.mojo}"
    local temp_dir="./temp"
    local perf_output="$temp_dir/out.perf"
    local flamegraph_output="flamegraph.svg"

    echo "Profiling $mojo_file..."

    if [ ! -d "$temp_dir" ]; then
        mkdir $temp_dir
    fi

    echo "Building $mojo_file..."
    mojo build -O0 -I . $mojo_file

    echo "Stripping debug symbols..."
    mv "$mojo_name" "$temp_dir/run.exe"
    llvm-strip --strip-debug "$temp_dir/run.exe"

    echo "Running perf record..."
    ~/perf record -F 99 -a -g -o $perf_output -- $temp_dir/run.exe

    echo "Generating flamegraph..."
    ~/perf script -i $perf_output | ~/FlameGraph/stackcollapse-perf.pl | ~/FlameGraph/flamegraph.pl > $flamegraph_output

    echo "Opening flamegraph: $flamegraph_output"
    explorer.exe $flamegraph_output

    echo "Cleaning up temporary files..."
    rm -rf $temp_dir

    echo "Profiling completed."
}

profile "$1"