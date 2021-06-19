/* eslint-disable */
// let net;

// CHANGE this url eventually
let model = null;

async function app() {
  model = await tf.loadLayersModel("http://127.0.0.1:5500/model.json");
  model.compile({
    optimizer: "adam",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });
}

app();

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

const classNames = [
  "A",
  "B",
  "C",
  "D",
  "E",
  "F",
  "G",
  "H",
  "I",
  "J",
  "K",
  "L",
  "M",
  "N",
  "O",
  "P",
  "Q",
  "R",
  "S",
  "T",
  "U",
  "V",
  "W",
  "X",
  "Y",
  "Z",
];

document.getElementById("butt").addEventListener("click", makePrediction);

document.getElementById("correct").addEventListener("click", addCorrect);
document.getElementById("wrong").addEventListener("click", addWrong);

let tensor;
let index;

// https://enlight.nyc/projects/web-paint
const srcCanvas = document.getElementById("src");
const destCanvas = document.getElementById("dest");

const srcCtx = $("#src")[0].getContext("2d");
const destCtx = $("#dest")[0].getContext("2d");

// initialize position as 0,0
const pos = { x: 0, y: 0 };

// new position from mouse events https://stackoverflow.com/questions/3234256/find-mouse-position-relative-to-element/42111623#42111623
// BUG: clicking dots (position doesn't change or is same) doesn't work
function setPosition(e) {
  const rect = e.target.getBoundingClientRect();
  pos.x = e.clientX - rect.left;
  pos.y = e.clientY - rect.top;
}

function draw(e) {
  if (e.buttons !== 1) return; // if mouse is not clicked, do not go further

  // var color = document.getElementById("hex").value;

  srcCtx.beginPath(); // begin the drawing path

  srcCtx.lineWidth = 35; // width of line
  srcCtx.lineCap = "round"; // rounded end cap
  // ctx.strokeStyle = color; // hex color of line

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
  // destCtx.save();
  destCtx.setTransform(1, 0, 0, 1, 0, 0);
  destCtx.clearRect(0, 0, destCanvas.width, destCanvas.height);
  // destCtx.restore();
  srcCtx.clearRect(0, 0, srcCanvas.width, srcCanvas.height);
  index = null;
  tensor = null;
  document.getElementById("pred").innerHTML = "Predictions: ";
}

// Used from http://jsfiddle.net/Hm2xq/2/ + https://stackoverflow.com/questions/3448347/how-to-scale-an-imagedata-in-html-canvas/3449416#3449416
function resize() {
  const imageData = srcCtx.getImageData(
    0,
    0,
    srcCanvas.width,
    srcCanvas.height,
  );

  const newCanvas = $("<canvas>").attr("width", 560).attr("height", 560)[0];
  newCanvas.getContext("2d").putImageData(imageData, 0, 0);
  destCtx.scale(0.05, 0.05);
  destCtx.drawImage(newCanvas, 0, 0);

  const destImageData = destCtx.getImageData(
    0,
    0,
    destCanvas.width,
    destCanvas.height
  );
  // imagedata object: r g b a (0, 1, 2 ,3) for first pixel
  // 28 x 28 = 784 pixels so 784 x 4 = 3136 - starting from 0 index so 3135

  return destImageData;
}

function makePrediction() {
  const destImageData = resize();
  tensor = tf.tensor(processImgData(destImageData.data), [1, 784]);
  const prediction = model.predict(tensor);
  const jsArray = prediction.dataSync();

  const max = Math.max(...jsArray);
  for (let i = 0; i < 25; i++) {
    if (jsArray[i] == max) {
      index = i;
    }
  }

  // console.log(tf.argMax(prediction, 0).print()); <-- figure out why this no work
  document.getElementById(
    "pred"
  ).innerHTML = `Predictions: ${classNames[index]}`;

  // test_loss, test_acc = model.evaluate(newTest,  test_labels, verbose=2)
  // print('\nTest accuracy:', test_acc);
}
function onBatchEnd(batch, logs) {
  console.log("Accuracy", logs.acc);
}

async function addCorrect() {
  const label = new Array(26).fill(0);
  label[index] = 1;

  model
    .fit(tensor, tf.tensor(label, [1, 26]), {
      epochs: 10,
      callbacks: { onBatchEnd },
    })
    .then((info) => {
      console.log("Final Accuracy", info.history.acc);
    });
  await model.save("localstorage://model");

  eraseFunction();
}

async function addWrong() {
  // wrong answer get input
  console.log(index);
  const wrongChar = document.getElementById("input").value;
  console.log(wrongChar);

  for (let i = 0; i < 26; i += 1) {
    if (wrongChar === classNames[i]) {
      index = i;
      break;
    }
  }

  const label = new Array(26).fill(0);
  label[index] = 1;

  model
    .fit(tensor, tf.tensor(label, [1, 26]), {
      epochs: 10,
      callbacks: { onBatchEnd },
    })
    .then((info) => {
      console.log("Final Accuracy", info.history.acc);
    });
  await model.save("localstorage://model");
}

// advice from: https://stackoverflow.com/questions/17945972/converting-rgba-values-into-one-integer-in-javascript
function revertRGBA(red, green, blue, alpha) {
  const r = red & 0xff;
  const g = green & 0xff;
  const b = blue & 0xff;
  const a = alpha & 0xff;

  // eslint-disable-next-line no-bitwise
  let rgb = (r << 24) + (g << 16) + (b << 8) + a;
  rgb /= 255.0;
  return rgb;
}

function processImgData(imgData) {
  const imgArray = []; // want a 784-long array
  for (let i = 1; i <= 784; i += 1) {
    const red = imgData[4 * (i - 1)];
    const green = imgData[1 + 4 * (i - 1)];
    const blue = imgData[2 + 4 * (i - 1)];
    const alpha = imgData[3 + 4 * (i - 1)];
    imgArray.push(revertRGBA(red, green, blue, alpha));
  }

  return imgArray;
}
