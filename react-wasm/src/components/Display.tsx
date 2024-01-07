import { useEffect } from "react"
import init, { inference, train } from "../../pkg/candle_wasm.js";
import * as ort from 'onnxruntime-web';

export default function Display() {
  useEffect(() => {
    init().then(() => {
      // console.log(add(1, 2));
      train();
      // console.log(inference())
    });

    async function main() {
      const session = await ort.InferenceSession.create("nn.onnx");
      const data = Float32Array.from([2.]);
      const tensor = new ort.Tensor('float32', data, [1, 1]);

      const feeds = { input: tensor };
      const results = await session.run(feeds);
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