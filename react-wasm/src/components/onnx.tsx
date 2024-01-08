import * as ort from 'onnxruntime-web';
import React from "react"

import Button from "./button"

type OrtInferenceSession = ort.InferenceSession | null


export default function Onnx() {
  const [session, setSession] = React.useState<OrtInferenceSession>(null)
  const [lastClicked, setLastClicked] = React.useState<number | null>(null)
  const [weights, setWeights] = React.useState([
    {"id": 0, "weight": null},
    {"id": 1, "weight": null},
    {"id": 2, "weight": null},
    {"id": 3, "weight": null},
    {"id": 4, "weight": null},
    {"id": 5, "weight": null},
  ])

  React.useEffect(() => {
    async function main() {
      const newSession = await ort.InferenceSession.create("nn.onnx");
      setSession(newSession);
    }
    main();
  }, [])

  function updateLastClicked(id: number) {
    setLastClicked(id)
  }

  function inference(input: number) {
    (async () => {
      const data = Float32Array.from([input]);
      const tensor = new ort.Tensor('float32', data, [1, 1]);

      const feeds = { input: tensor };
      if (session !== null) {
        const results = await session.run(feeds);

        const newWeights = (
          Array.from(results.output.data)
          .map((weight, index) => ({"id": index, "weight": weight}))
          .sort((a, b) => b.weight - a.weight)
        )
        setWeights(newWeights)
      }
    })()
  }

  return (
    <div>
      <div className='last-clicked'>
        Last clicked: {lastClicked}
      </div>
      <div className='buttons'>
        {weights.map((i) => <Button key={i.id} value={i.id} weight={i.weight} handleClick={(() => {
          inference(i.id);
          updateLastClicked(i.id);
          })} />)}
      </div>
    </div>
  )
}