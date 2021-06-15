let net;

// CHANGE this url eventually
const model = await tf.loadLayersModel('http://127.0.0.1:5500/model.json');

// async function app() {
//   console.log('Loading mobilenet..');
//   // Load the model.
//   net = await mobilenet.load();
//   console.log('Successfully loaded model');

//   // Make a prediction through the model on our image.
//   const imgEl = document.getElementById('img');
//   const result = await net.classify(imgEl);
//   console.log(result);
// }

const class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 
'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'];

document.getElementById("butt").addEventListener("click", resize);

function clickFunction() {
  // preprocess image to make it 28 by 28, grey scale
  // run into model
  // const example = tf.fromPixels(webcamElement);  // for example
  // const prediction = model.predict(example);
  // display predictions on screen
  document.getElementById("pred").innerHTML = "Change to prediction!!!";
}

// https://enlight.nyc/projects/web-paint
var srcCanvas = document.getElementById("src");
var destCanvas = document.getElementById("dest");


var srcCtx = $("#src")[0].getContext("2d");
var destCtx = $("#dest")[0].getContext("2d");


//resize();

// // resize canvas when window is resized
// function resize() {
//   ctx.canvas.width = window.innerWidth;
//   ctx.canvas.height = window.innerHeight;
// }

// initialize position as 0,0
var pos = { x: 0, y: 0 };

// new position from mouse events
function setPosition(e) {
  pos.x = e.clientX;
  pos.y = e.clientY;
}

function draw(e) {
  if (e.buttons !== 1) return; // if mouse is not clicked, do not go further

  //var color = document.getElementById("hex").value;

  srcCtx.beginPath(); // begin the drawing path

  srcCtx.lineWidth = 20; // width of line
  srcCtx.lineCap = "round"; // rounded end cap
  //ctx.strokeStyle = color; // hex color of line

  srcCtx.moveTo(pos.x, pos.y); // from position
  setPosition(e);
  srcCtx.lineTo(pos.x, pos.y); // to position

  srcCtx.stroke(); // draw it!
}


// add window event listener to trigger when window is resized
window.addEventListener("resize", resize);

// add event listeners to trigger on different mouse events
document.addEventListener("mousemove", draw);
document.addEventListener("mousedown", setPosition);
document.addEventListener("mouseenter", setPosition);

document.getElementById("erase").addEventListener("click", eraseFunction);

function eraseFunction() {
  // transformed coordinates: https://stackoverflow.com/questions/2142535/how-to-clear-the-canvas-for-redrawing
  destCtx.save();
  destCtx.setTransform(1, 0, 0, 1, 0, 0);
  destCtx.clearRect(0, 0, destCanvas.width, destCanvas.height);
  //destCtx.restore();
  srcCtx.clearRect(0, 0, srcCanvas.width, srcCanvas.height);
  
}

// Used from http://jsfiddle.net/Hm2xq/2/ + https://stackoverflow.com/questions/3448347/how-to-scale-an-imagedata-in-html-canvas/3449416#3449416
function resize() {
  var imageData = srcCtx.getImageData(0, 0, srcCanvas.width, srcCanvas.height)
  var newCanvas = $("<canvas>").attr("width", 560)
.attr("height", 560)[0];
  newCanvas.getContext("2d").putImageData(imageData, 0, 0);
  destCtx.scale(0.05, 0.05);
  destCtx.drawImage(newCanvas, 0, 0);

  var destImageData = destCtx.getImageData(0, 0, destCanvas.width, destCanvas.height);
  // imagedata object: r g b a (0, 1, 2 ,3) for first pixel
  // 28 x 28 = 784 pixels so 784 x 4 = 3136 - starting from 0 index so 3135

  var tensor = tf.tensor(processImgData(destImageData), [1, 784]);
  var prediction = model.predict(tensor);
  console.log(prediction);

  prediction.print();
 
  var jsArray = prediction.dataSync();
 
  const max = Math.max(...jsArray);
  console.log("Max ", max);
  var index;
  for (var i = 0; i < 25; i++) {
    if (jsArray[i] == max) {
      index = i;
    }
  }
  // console.log(tf.argMax(prediction, 0).print()); <-- figure out why this no work
  console.log(index);
  console.log(class_names[index]);
  
 
 
  

}

function getPrediction() {
  
}

// advice from: https://stackoverflow.com/questions/17945972/converting-rgba-values-into-one-integer-in-javascript
function revertRGBA(red, green, blue, alpha) {
  var r = red & 0xFF;
  var g = green & 0xFF;
  var b = blue & 0xFF;
  var a = alpha & 0xFF;

  var rgb = (r << 24) + (g << 16) + (b << 8) + (a);   
  return rgb;
}

function processImgData(imgData) {
  var imgArray = []; // want a 784-long array
  for(var i = 1; i <= 784; i++) {
    var red = imgData[4 * (i - 1)];
    var green = imgData[1 + 4 * (i - 1)];
    var blue = imgData[2 + 4 * (i - 1)];
    var alpha = imgData[3 + 4 * (i - 1)];
    imgArray.push(revertRGBA(red, green, blue, alpha));
  }
  return imgArray;

}




