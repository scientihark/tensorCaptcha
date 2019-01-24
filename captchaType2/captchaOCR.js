const {PythonShell} = require('python-shell');

const PY_SCRIPT = 'predict.py'

module.exports = (file,successCB,errCB)=>{
	PythonShell.run(PY_SCRIPT, {
		args:['--fname', file],
		pythonPath:'python',
	}, (err, results) => {
	    if (err || !results || !results[0] || !/[0-9A-Za-z]{4}/.test(results[0]) ||results[0].length != 4 ) {
	    	errCB && errCB(results)
	    } else {
	    	successCB(results[0])
	    }
	})
}

