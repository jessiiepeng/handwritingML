let net;

// CHANGE this url eventually
const model = await tf.loadLayersModel('http://127.0.0.1:5500/model.json');

async function app() {
  console.log('Loading mobilenet..');
  // Load the model.
  net = await mobilenet.load();
  console.log('Successfully loaded model');

  // Make a prediction through the model on our image.
  const imgEl = document.getElementById('img');
  const result = await net.classify(imgEl);
  console.log(result);
}


document.getElementById("butt").addEventListener("click", clickFunction);

function clickFunction() {
  // preprocess image to make it 28 by 28, grey scale
  // run into model
  // const example = tf.fromPixels(webcamElement);  // for example
  // const prediction = model.predict(example);
  // display predictions on screen
  document.getElementById("pred").innerHTML = "Change to prediction!!!";
}

app();