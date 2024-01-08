import './App.css'
import Onnx from './components/onnx';
import Candle from './components/candle';


export default function App() {
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