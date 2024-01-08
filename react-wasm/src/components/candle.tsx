import React from "react"
import init, { Model } from "../../pkg/candle_wasm"

import Button from "./button"

export default function Candle() {
  const [session, setSession] = React.useState<object | null>(null);
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
    init().then(() => {
      const model = new Model();
      setSession(model);
    //   console.log(model.train(2., 1));
  });
  }, [])

  function inference(id: number) {
    if (session === null) {
      console.log("session not initialized")
      return
    }

    if (lastClicked === null) {
      console.log("No last clicked item yet")
      return
    }

    const results = session.train(lastClicked, id)
    const newWeights = (
      Array.from(results)
      .map((weight, index) => ({"id": index, "weight": weight}))
      .sort((a, b) => b.weight - a.weight)
    )
    setWeights(newWeights)
  }


  function updateLastClicked(id: number) {
    setLastClicked(id)
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