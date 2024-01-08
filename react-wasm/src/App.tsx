import * as ort from 'onnxruntime-web';
// import init, { train, Model } from "../../pkg/candle_wasm.js";
import React from "react"
import Onnx from './components/onnx';
import './App.css'
import Candle from './components/candle';


type OrtInferenceSession = ort.InferenceSession | null


export default function App() {
    React.useEffect(() => {
      // init().then(() => {
      //   const model = new Model();
      //   console.log(model.train(2., 1));
      //   console.log(model.train(2., 1));
      //   console.log(model.train(2., 1));
      //   // console.log(inference_onnx())
      // });
    }, [])

  return (
    <div>
      <h1> WASM </h1>
      <div>
        <h3>ONNX</h3>
        <Onnx />
      </div>
      <div>
        <h3>CANDLE</h3>
        <Candle />
      </div>
    </div>
  )
}