#!/bin/bash

# Make sure node version is greater than 15.1.0
ver="$(node --version)"
ver=${ver:1} # remove the prefix 'v'
major="$(cut -d '.' -f 1 <<< "$ver")"
minor="$(cut -d '.' -f 2 <<< "$ver")"

if [ $major -ge 15 ] && [ $minor -ge 1 ]; then
    node --trace-uncaught --experimental-wasm-threads script.js
else
    echo "Cannot run node. Require version 15.1.0 or greater"
fi
