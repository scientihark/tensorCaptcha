const fs = require('fs');
const { spawn } = require('child_process');

const MAP = {};

let list = fs.readdirSync('testSample'),
	lock = false;

try{
	fs.mkdirSync('data');
}catch(e){}

try{
	fs.mkdirSync('data/captcha');
}catch(e){}

function preProcessImg(){
	
	if(lock){
		return;
	}
	lock = true;
	
	let filePath = list.pop();
	if(!filePath){
		clearInterval(timer);
		clearInterval(timer2);
		fs.writeFileSync('data/captcha.json',JSON.stringify(MAP));
		return;
	}

	const magick_1 = spawn('magick', ['convert', 'testSample/'+filePath,'-density','300','-units','PixelsPerInch','-type','Grayscale','-negate','-scale','200%','-morphology','Erode','Octagon:1',
						   '-negate','-morphology', 'Erode', 'disk:1','-scale','50%','-threshold','99%','-colors','2','-colorspace','gray','-background','white','-flatten','data/captcha/'+filePath+'.png']);

	magick_1.stderr.on('data', (data) => {
	  console.log(`magick_1 stderr: ${data}`);
	});

	magick_1.on('close', (code) => {
		if(code){
			console.log(code)
		} else {
			MAP[filePath+'.png'] = filePath.split('.')[0];
		}
		lock = false;
	});
}

var timer = setInterval(preProcessImg,1);
var timer2 = setInterval(()=>{
	console.log(list.length)
},1000);



