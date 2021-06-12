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


document.getElementById("butt").addEventListener("click", clickFunction);

function clickFunction() {
  // preprocess image to make it 28 by 28, grey scale
  // run into model
  // const example = tf.fromPixels(webcamElement);  // for example
  // const prediction = model.predict(example);
  // display predictions on screen
  document.getElementById("pred").innerHTML = "Change to prediction!!!";
}

// https://enlight.nyc/projects/web-paint
var canvas = document.getElementById("canvas");

var ctx = canvas.getContext("2d");
//resize();

// resize canvas when window is resized
function resize() {
  ctx.canvas.width = window.innerWidth;
  ctx.canvas.height = window.innerHeight;
}

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

  ctx.beginPath(); // begin the drawing path

  ctx.lineWidth = 20; // width of line
  ctx.lineCap = "round"; // rounded end cap
  //ctx.strokeStyle = color; // hex color of line

  ctx.moveTo(pos.x, pos.y); // from position
  setPosition(e);
  ctx.lineTo(pos.x, pos.y); // to position

  ctx.stroke(); // draw it!
}


// add window event listener to trigger when window is resized
window.addEventListener("resize", resize);

// add event listeners to trigger on different mouse events
document.addEventListener("mousemove", draw);
document.addEventListener("mousedown", setPosition);
document.addEventListener("mouseenter", setPosition);

document.getElementById("erase").addEventListener("click", eraseFunction);
function eraseFunction() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}