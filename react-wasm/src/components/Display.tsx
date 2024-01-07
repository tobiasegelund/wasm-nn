import { useEffect } from "react"
import init, { train, Model } from "../../pkg/candle_wasm.js";
import * as ort from 'onnxruntime-web';

export default function Display() {
  useEffect(() => {
    init().then(() => {
      // console.log(add(1, 2));
      const model = new Model();
      console.log(model.train(2., 1));
      console.log(model.train(2., 1));
      console.log(model.train(2., 1));
      // console.log(train(2., 1));
      // console.log(inference_onnx())
    });

    async function main() {
      const session = await ort.InferenceSession.create("nn.onnx");
      const data = Float32Array.from([2.]);
      const tensor = new ort.Tensor('float32', data, [1, 1]);

      const feeds = { input: tensor };
      const results = await session.run(feeds);
      // console.log(results.output.data)
    }
    main();

  }, [])

  return (
    <div>
      <h1> WASM </h1>
    </div>
  )
}