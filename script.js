// ----------------------------------------
// To be used with node.js
// ----------------------------------------

const LibModule = require('./lib')

function runMain(method) {
    LibModule().then( lib => {
        const a = performance.now()
        if (method === 'thread') {
            lib.thread()
        }
        if (method === 'serial') {
            lib.serial()
        }
        if (method === 'mutex') {
            lib.mutex()
        }
        if (method === 'go') {
            lib.go()
        }
        
        console.log( `${method} done in ${(performance.now()-a)}ms` )
    })
}

// runMain('thread')
// runMain('serial')
runMain('go')
