// ----------------------------------------
// To be used with node.js
// ----------------------------------------

const LibModule = require('./lib')

function runMain() {
    LibModule().then( lib => {
        const a = performance.now()
        lib.go()
    })
}

runMain()
