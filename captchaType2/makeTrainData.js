const fs = require('fs');
const { spawn } = require('child_process');

let MAP = {};

let list = fs.readdirSync('testSample'),
	lock = false;

try{
	fs.mkdirSync('data');
}catch(e){}

try{
	fs.mkdirSync('data/captcha');
}catch(e){}

try{
	MAP = JSON.parse(fs.readFileSync('data/captcha.json','utf-8'))
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
	fs.copyFileSync(`testSample/${filePath}`, `data/captcha/${filePath}`)
	MAP[filePath] = filePath.split('.')[0];
	lock = false;
}

var timer = setInterval(preProcessImg,1);
var timer2 = setInterval(()=>{
	console.log(list.length)
},1000);



