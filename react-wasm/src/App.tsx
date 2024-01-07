import * as ort from 'onnxruntime-web';
// import init, { train, Model } from "../../pkg/candle_wasm.js";
import React from "react"
import Button from './components/Button';
import './App.css'


export default function App() {
  const [session, setSession] = React.useState<ort.InferenceSession | null>(null)
  const labels = [0, 1, 2, 3, 4, 5]

  React.useEffect(() => {
    // init().then(() => {
    //   const model = new Model();
    //   console.log(model.train(2., 1));
    //   console.log(model.train(2., 1));
    //   console.log(model.train(2., 1));
    //   // console.log(inference_onnx())
    // });

    async function main() {
      const newSession = await ort.InferenceSession.create("nn.onnx");
      setSession(newSession);
    }
    main();
  }, [])

  function inference(input: number) {
    (async () => {
      const data = Float32Array.from([input]);
      const tensor = new ort.Tensor('float32', data, [1, 1]);

      const feeds = { input: tensor };
      try {
        const results = await session.run(feeds);
        const output = results.output.data
        console.log(output)
      } catch (err) {
        console.error("Error during inference:", err);
      }
    })()
  }

  return (
    <div>
      <h1> WASM </h1>
      <div className='buttons'>
        {labels.map((i) => <Button value={i} handleClick={() => inference(i)} />)}
      </div>
    </div>
  )
}