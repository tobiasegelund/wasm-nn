import { useEffect } from "react"
import init, { inference, add } from "../../pkg/burn_wasm.js";

export default function Display() {
  useEffect(() => {
    init().then(() => {
      console.log(add(1, 2));
      console.log(inference());
      console.log(inference());
      // console.log(train());
    });

  }, [])

  return (
    <div>
      <h1> WASM </h1>
    </div>
  )
}