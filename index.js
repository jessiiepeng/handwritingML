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

document.getElementById("butt").addEventListener("click", makePrediction);


// https://enlight.nyc/projects/web-paint
var srcCanvas = document.getElementById("src");
var destCanvas = document.getElementById("dest");

var srcCtx = $("#src")[0].getContext("2d");
var destCtx = $("#dest")[0].getContext("2d");

// initialize position as 0,0
var pos = { x: 0, y: 0 };

// new position from mouse events https://stackoverflow.com/questions/3234256/find-mouse-position-relative-to-element/42111623#42111623
// BUG: clicking dots (position doesn't change or is same) doesn't work
function setPosition(e) {
  var rect = e.target.getBoundingClientRect();
  pos.x = e.clientX - rect.left;
  pos.y = e.clientY - rect.top;
}

function draw(e) {
  if (e.buttons !== 1) return; // if mouse is not clicked, do not go further

  //var color = document.getElementById("hex").value;

  srcCtx.beginPath(); // begin the drawing path

  srcCtx.lineWidth = 35; // width of line
  srcCtx.lineCap = "round"; // rounded end cap
  //ctx.strokeStyle = color; // hex color of line

  srcCtx.moveTo(pos.x, pos.y); // from position
  setPosition(e);
  srcCtx.lineTo(pos.x, pos.y); // to position

  srcCtx.stroke(); // draw it!
}


// add event listeners to trigger on different mouse events
document.addEventListener("mousemove", draw);
document.addEventListener("mousedown", setPosition);
document.addEventListener("mouseenter", setPosition);

document.getElementById("erase").addEventListener("click", eraseFunction);

function eraseFunction() {
  // transformed coordinates: https://stackoverflow.com/questions/2142535/how-to-clear-the-canvas-for-redrawing
  //destCtx.save();
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

  return destImageData;

}

function makePrediction() {

  var destImageData = resize();
  var tensor = tf.tensor(processImgData(destImageData.data), [1, 784]);
  var prediction = model.predict(tensor);
  var jsArray = prediction.dataSync();
 
  const max = Math.max(...jsArray);
  var index;
  for (var i = 0; i < 25; i++) {
    if (jsArray[i] == max) {
      index = i;
    }
  }
  // console.log(tf.argMax(prediction, 0).print()); <-- figure out why this no work
  console.log(class_names[index]);
  document.getElementById("pred").innerHTML = class_names[index];
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

// advice from: https://stackoverflow.com/questions/17945972/converting-rgba-values-into-one-integer-in-javascript
function revertRGBA(red, green, blue, alpha) {
  var r = red & 0xFF;
  var g = green & 0xFF;
  var b = blue & 0xFF;
  var a = alpha & 0xFF;

  var rgb = (r << 24) + (g << 16) + (b << 8) + (a);   
  rgb = rgb / 255.0;
  return rgb;
}



