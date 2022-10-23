// ----------------------------------------
// To be used with node.js
// ----------------------------------------

const LibModule = require('./lib')

function runMain() {
    LibModule().then( lib => {
        let a = performance.now()
        lib.testSerial()
        console.log( `serial done in ${(performance.now()-a)}ms` )

        a = performance.now()
        lib.testMpi()
        console.log( `MPI done in ${(performance.now()-a)}ms` )
    })
}

runMain()
