const { spawn } = require('child_process');
const {PythonShell} = require('python-shell');

const PY_SCRIPT = 'predict.py'

function preProcessImg(filePath,successCB,errorCB){

	const magick_1 = spawn('magick', ['convert', filePath,'-density','300','-units','PixelsPerInch','-type','Grayscale','-negate','-scale','200%','-morphology','Erode','Octagon:1',
							'-negate','-morphology', 'Erode', 'disk:1','-scale','50%','-threshold','99%','-colors','2','-colorspace','gray','-background','white','-flatten',filePath+'.tmp.png']);

	const errorMsg = null;
	magick_1.stderr.on('data', (data) => {
	  	errorMsg = data;
	});

	magick_1.on('close', (code) => {
		if(code || errorMsg){
			errorCB && errorCB(errorMsg)
		} else {
			successCB && successCB()
		}
	});
}

module.exports = (file,successCB,errCB)=>{
	preProcessImg(file,()=>{
		PythonShell.run(PY_SCRIPT, {
			args:['--fname', file+'.tmp.png'],
			pythonPath:'python',
		}, (err, results) => {
		    if (err || !results || !results[0] || !/[0-9]{4}/.test(results[0])) {
		    	errCB && errCB(results)
		    } else {
		    	successCB(results[0])
		    }
		})
	},errCB)
}

