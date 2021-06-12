let net;
import * as tf from '@tensorflow/tfjs';

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

app();