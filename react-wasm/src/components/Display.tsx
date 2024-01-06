import { useEffect } from "react"
// import init, { inference } from "../../pkg/burn_wasm.js";
import init, { inference, train } from "../../pkg_candle/candle_wasm.js";
import * as ort from 'onnxruntime-web';

export default function Display() {
  useEffect(() => {
    init().then(() => {
      // console.log(add(1, 2));
      console.log(train());
      // console.log(train());
    });

    async function main() {
      const session = await ort.InferenceSession.create("nn.onnx");
      const data = Float32Array.from([2.]);
      const tensor = new ort.Tensor('float32', data, [1, 1]);

      const feeds = { input: tensor };
      const start = new Date();
      const results = await session.run(feeds);
      const end = new Date();
      console.log(end.getTime() - start.getTime())
      console.log(results.output.data)
    }
    main();

  }, [])

  return (
    <div>
      <h1> WASM </h1>
    </div>
  )
}