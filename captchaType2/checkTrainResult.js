const fs = require('fs');
const { spawn } = require('child_process');
const OCR = require('./captchaOCR.js');

const DIR = 'testSample'

let list = fs.readdirSync(DIR),
	lock = false
	successed = 0,
	total = 0;

function preProcessImg(){
	if(lock){
		return;
	}
	lock = true;

	let filePath = list.pop();
	if(!filePath){
		clearInterval(timer);
		clearInterval(timer2);
		console.log(list.length,total,successed,Math.floor((successed/total)*100000000)/1000000)
		return;
	}

	OCR(`${DIR}/${filePath}`,(str)=>{

		let trueResult = filePath.split('.')[0];
		if(str && str == trueResult){
			successed ++
		}

		total ++;
		lock = false;

	},()=>{

		total ++;
		lock = false;
		
	})
	
}

var timer = setInterval(preProcessImg,1);
var timer2 = setInterval(()=>{
	console.log(list.length,total,successed,Math.floor((successed/total)*100000000)/1000000)
},1000);



